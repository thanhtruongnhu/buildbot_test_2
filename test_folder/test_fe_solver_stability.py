# Copyright (C) 2015 Akselos
import os
import redis
import uuid
import netCDF4
import tempfile

import set_akselos_path  # noqa
import akselos.utils
import directories
import json_helper
import interfaces.akselos_assembly
import interfaces.fe_solver_args
import solver_options
import server.run_exe
import types_data
import test_rb_solver_stability
import dimensional_value as dv
import math
import train.job_options as jo


def vec3d_norm(x, y, z):
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def get_solver_response(job_id):
    redis_server = redis.Redis("localhost")
    return redis_server.hget("job:{}".format(job_id), "result")


def decompress_file(file_path):
    if file_path.endswith('.bz2') or file_path.endswith('.gz'):
        _, filename = os.path.split(file_path)
        with open(file_path, 'rb') as f:
            data = f.read()

        data, filename = akselos.utils.decompress_data(data, filename)

        temp_gmv_file = tempfile.NamedTemporaryFile(delete=False)
        temp_gmv_file.write(data)
        temp_gmv_file.close()
        return temp_gmv_file.name
    else:
        return file_path


def compare_exo_files(exo1, exo2, tol):
    use_exo1 = use_exo2 = None
    try:
        use_exo1 = decompress_file(exo1)
        use_exo2 = decompress_file(exo2)
        loaded1 = netCDF4.Dataset(use_exo1)
        loaded2 = netCDF4.Dataset(use_exo2)
        test_rb_solver_stability.compare(loaded1, loaded2, tol)
    finally:
        if use_exo1 is not None and use_exo1 != exo1:
            os.unlink(use_exo1)

        if use_exo2 is not None and use_exo2 != exo2:
            os.unlink(use_exo2)


def test_fe_solver(aks_filename,
                   reference_solution_exo_filename,
                   num_cores,
                   mesh_stitching_tolerance,
                   tolerance,
                   use_elem_stress,
                   von_mises_stress_only,
                   use_global_coords_for_surface_constraints,
                   is_shell,
                   do_not_compare_exo_files,
                   solve_type,
                   uncoupled_submodel_solve,
                   submodel_ids,
                   time_parameters,
                   contact_algorithm,
                   finite_sliding,
                   ref_x, ref_y, ref_z,
                   ref_xy, ref_xz, ref_yz,
                   origin_x, origin_y, origin_z,
                   force_moment_test_type, ref_units,
                   skip_precomputed_interface_functions,
                   include_materials_in_plot,
                   max_dofs_per_port=-1,
                   allow_nonconforming_port_meshes=False,
                   use_reduced_integration=False,
                   use_parallel_mesh_stitching=False,
                   compare_component_solutions=False,
                   do_not_use_component_datasets=False):
    stitching_tolerance = mesh_stitching_tolerance
    tol = tolerance

    types = types_data.TypesData()

    akselos_assembly = json_helper.read(interfaces.akselos_assembly.AkselosAssembly,
                                        open(aks_filename).read())

    collection_name = akselos_assembly.component_system.collection_type
    collection = types.get_collection(collection_name)
    assert collection is not None, 'Collection not found: {}'.format(collection_name)
    physics_name = collection.get_physics_name().upper()

    is_hybrid_solve = False
    if submodel_ids is not None:
        is_hybrid_solve = len(submodel_ids) > 0

    compute_stress = physics_name == "ELASTICITY"

    solver_parameters = interfaces.component_system_data.SolverOptionsData()
    solver_parameters.real_valued_data_present = True
    solver_parameters.string_data_present = True
    solver_parameters.int_valued_data_present = True
    solver_parameters.boolean_data_present = True

    if is_hybrid_solve:
        solver_parameters.solver_strategy = "hybrid"
    else:
        solver_parameters.solver_strategy = "fea"

    solver_parameters.string_data["linear_solver"] = "Direct (MUMPS)"
    solver_parameters.string_data["contact_solver_type"] = contact_algorithm

    solver_parameters.boolean_data["allow_nonconforming_port_meshes"] = allow_nonconforming_port_meshes
    solver_parameters.boolean_data["use_global_coords_for_surface_constraints"] = \
      use_global_coords_for_surface_constraints
    solver_parameters.boolean_data["skip_precomputed_interface_functions"] = \
      skip_precomputed_interface_functions
    solver_parameters.boolean_data["is_finite_sliding"] = finite_sliding

    # Optionally set reduced integration flag
    if use_reduced_integration:
        solver_parameters.boolean_data["use_reduced_integration"] = True

    # Optionally set max_dofs_per_port. This is relevant for Hybrid solves
    # (it does not affect full FEA solves).
    if max_dofs_per_port >= 0:
        solver_parameters.int_valued_data["max_dofs_per_port"] = max_dofs_per_port

    # Possibly skip component datasets (relevant in Hybrid solves)
    solver_parameters.boolean_data["use_component_datasets"] = not do_not_use_component_datasets

    # Parallel mesh stitching
    solver_parameters.boolean_data["use_parallel_mesh_stitching"] = use_parallel_mesh_stitching

    if solve_type == "contact" or solve_type == "frictional_contact":
        if is_hybrid_solve:
            solver_parameters.solver_type = "hybrid_linear_elasticity_contact"
        else:
            solver_parameters.solver_type = "fe_linear_elasticity_contact"

        solver_parameters.real_valued_data["Nonlinear abs. tol."] = 1.e-12
        solver_parameters.real_valued_data["Nonlinear rel. tol."] = 1.e-6
        solver_parameters.real_valued_data["Gap/overlap tolerance"] = 1.e-6

        solver_parameters.int_valued_data["Max. inner nonlinear iterations"] = 100
        solver_parameters.int_valued_data["Max. outer iterations"] = 100

        solver_parameters.string_data["line_search_type"] = "Basic"
    elif solve_type == "plasticity":
        if is_hybrid_solve:
            solver_parameters.solver_type = "hybrid_equilibrium_plasticity"
        else:
            solver_parameters.solver_type = "fe_equilibrium_plasticity"

        solver_parameters.int_valued_data["Max. inner nonlinear iterations"] = 100

        solver_parameters.real_valued_data["Nonlinear abs. tol."] = 1.e-12
        solver_parameters.real_valued_data["Nonlinear rel. tol."] = 1.e-6
        solver_parameters.real_valued_data["Linear rel. tol."] = 1.e-6

        solver_parameters.string_data["line_search_type"] = "Backtracking"
    elif solve_type == "nlgeo":
        solver_parameters.solver_type = "fe_nlgeo_elasticity"

        solver_parameters.int_valued_data["Max. inner nonlinear iterations"] = 100

        solver_parameters.real_valued_data["Nonlinear abs. tol."] = 1.e-12
        solver_parameters.real_valued_data["Nonlinear rel. tol."] = 1.e-6
        solver_parameters.real_valued_data["Linear rel. tol."] = 1.e-6

        solver_parameters.string_data["line_search_type"] = "Basic"
    elif solve_type == "plasticity_contact" or solve_type == "plasticity_frictional_contact":
        if is_hybrid_solve:
            solver_parameters.solver_type = "hybrid_plasticity_contact"
        else:
            solver_parameters.solver_type = "fe_plasticity_contact"

        solver_parameters.real_valued_data["Nonlinear abs. tol."] = 1.e-12
        solver_parameters.real_valued_data["Nonlinear rel. tol."] = 1.e-6
        solver_parameters.real_valued_data["Gap/overlap tolerance"] = 1.e-6
        solver_parameters.real_valued_data["Linear rel. tol."] = 1.e-6

        solver_parameters.int_valued_data["Max. outer iterations"] = 100
        solver_parameters.int_valued_data["Max. inner nonlinear iterations"] = 100

        solver_parameters.string_data["line_search_type"] = "Backtracking"
    elif solve_type == "dynamic_implicit":
        solver_parameters.solver_type = "fe_linear_elasticity_dynamic_implicit"

        solver_parameters.real_valued_data["Final time"] = time_parameters[0]
        solver_parameters.real_valued_data["Plot every"] = time_parameters[1]
        solver_parameters.real_valued_data["Time step"] = time_parameters[2]
    elif solve_type == "dynamic_explicit":
        solver_parameters.solver_type = "fe_linear_elasticity_dynamic_explicit"

        solver_parameters.real_valued_data["Final time"] = time_parameters[0]
        solver_parameters.real_valued_data["Plot every"] = time_parameters[1]
        solver_parameters.real_valued_data["Time step"] = time_parameters[2]
    elif solve_type == "thermoelasticity":
        if is_hybrid_solve:
            solver_parameters.solver_type = "hybrid_thermoelasticity"
        else:
            solver_parameters.solver_type = "fe_thermoelasticity"
    elif solve_type == "thermoelasticity_contact" or solve_type == "thermoelasticity_frictional_contact":
        if is_hybrid_solve:
            solver_parameters.solver_type = "hybrid_thermoelasticity_contact"
        else:
            solver_parameters.solver_type = "fe_thermoelasticity_contact"

        solver_parameters.real_valued_data["Nonlinear abs. tol."] = 1.e-12
        solver_parameters.real_valued_data["Nonlinear rel. tol."] = 1.e-6
        solver_parameters.real_valued_data["Gap/overlap tolerance"] = 1.e-6

        solver_parameters.int_valued_data["Max. inner nonlinear iterations"] = 100
        solver_parameters.int_valued_data["Max. outer iterations"] = 100

        solver_parameters.string_data["line_search_type"] = "Basic"
    elif solve_type == "calculate_net_force_and_moment":
        solver_parameters.solver_type = "fe_elasticity_net_force_and_moment"

        solver_parameters.real_valued_data["origin_x"] = origin_x
        solver_parameters.real_valued_data["origin_y"] = origin_y
        solver_parameters.real_valued_data["origin_z"] = origin_z

        # set max_dofs_per_port = 0 to avoid an error related to unsiged ints
        solver_parameters.int_valued_data["max_dofs_per_port"] = 0
    elif solve_type == "calculate_weight_data":
        solver_parameters.solver_type = "fe_elasticity_weight_data"

        solver_parameters.real_valued_data["origin_x"] = origin_x
        solver_parameters.real_valued_data["origin_y"] = origin_y
        solver_parameters.real_valued_data["origin_z"] = origin_z

        # set max_dofs_per_port = 0 to avoid an error related to unsiged ints
        solver_parameters.int_valued_data["max_dofs_per_port"] = 0
    elif solve_type == "plot_elasticity_materials":
        solver_parameters.solver_type = "fe_elasticity_plot_materials"
    elif solve_type == "plot_thermoelasticity_materials":
        solver_parameters.solver_type = "fe_thermoelasticity_plot_materials"
    else:
        if is_hybrid_solve:
            solver_parameters.solver_type = "hybrid_elasticity_default"
        else:
            solver_parameters.solver_type = "fe_elasticity_default"

    if uncoupled_submodel_solve:
        solver_parameters.boolean_data["uncoupled_submodel_solve"] = True

    if is_shell:
        solver_parameters.boolean_data["assumed_stress"] = True

    viz_parameters = interfaces.visualization_parameters.VisualizationParameters()
    viz_parameters.boolean_data_present = True
    viz_parameters.boolean_data["write_slice_exodus_files"] = True
    viz_parameters.boolean_data["write_slice_gmv_files"] = False
    viz_parameters.boolean_data["write_solution_exodus_file"] = True
    viz_parameters.boolean_data["write_solution_gmv_file"] = False
    viz_parameters.boolean_data["extract_slices"] = True

    if include_materials_in_plot:
        viz_parameters.boolean_data["include_materials_in_plot"] = True

    # Tell the solver not to remove subdomains/sidesets. We do this in
    # some cases in order to make writing/reading faster.
    viz_parameters.boolean_data["keep_subdomains_and_sidesets"] = True

    if compute_stress:
        viz_parameters.extra_fields_present = True
        if solve_type == "plasticity" or \
             solve_type == "plasticity_contact" or \
             solve_type == "plasticity_frictional_contact":
            viz_parameters.extra_fields.append("strain_tensor")
            viz_parameters.extra_fields.append("plastic_equivalent_strain")
            viz_parameters.extra_fields.append("stress_tensor")
            viz_parameters.extra_fields.append("von_mises")
            viz_parameters.extra_fields.append("elastic_strain")
            viz_parameters.extra_fields.append("plastic_strain")
            viz_parameters.extra_fields.append("principal_stresses")
        else:
            # If the user requested Von Mises stress only, just give
            # them that instead of all the stress and strain
            # components.
            if von_mises_stress_only:
                viz_parameters.extra_fields.append("von_mises")
            else:
                viz_parameters.extra_fields.append("strain_tensor")
                viz_parameters.extra_fields.append("stress_tensor")
    if use_elem_stress:
        viz_parameters.boolean_data["use_element_stress"] = True

    solver_parameters.num_cores = num_cores
    akselos_assembly.component_system.solver_options_data_present = True
    akselos_assembly.component_system.solver_options_data = [solver_parameters]

    fe_args = interfaces.fe_solver_args.FeSolverSolve(
        component_system=akselos_assembly.component_system,
        component_system_changes=akselos_assembly.component_system_changes_map.get('default', None),
        stitching_tolerance=stitching_tolerance,
        save_system=True,
        save_mesh_formats=[],
        visualization_parameters=viz_parameters,
        submodel_component_ids=submodel_ids)

    # We may receive two response values in the case that we are doing an RB-FE solve.  Here
    # we just take the FE response and ignore the RB repsonse. BP: run_scrbe_process is returning
    #  the first response. to have all, use run_scrbe_process_text
    job_options = jo.JobOptions(ppn=num_cores)
    fe_response = server.run_exe.run_scrbe_process(fe_args, job_options)

    if solve_type == "calculate_net_force_and_moment":
        force_moment_test_type_found = False
        for output_data in fe_response.output_datas:
            if output_data.output_type == force_moment_test_type:
                force_moment_test_type_found = True
                elements = []
                elements.append(output_data.real_valued_data["element_x"])
                elements.append(output_data.real_valued_data["element_y"])
                elements.append(output_data.real_valued_data["element_z"])


                if force_moment_test_type == "Net torque":
                    units = "moment"
                elif force_moment_test_type == "Net force":
                    units = "force"
                else:
                    assert False, "Invalid force_moment_test_type"

                _, encoded_elements, units_str = dv.encode_params(
                    collection, "", elements, units)

                computed_x = encoded_elements[0]
                computed_y = encoded_elements[1]
                computed_z = encoded_elements[2]

                print("Computed value: ({},{},{}) {}".format(
                    computed_x, computed_y, computed_z, units_str))

                error_x = abs(computed_x - ref_x)
                error_y = abs(computed_y - ref_y)
                error_z = abs(computed_z - ref_z)
                ref_norm = vec3d_norm(ref_x, ref_y, ref_z)

                # compare unit regardless of the order.
                assert set(units_str.split()) == set(ref_units.split()), \
                    "ERROR: Wrong units! {} vs. {}".format(ref_units, units_str)

                rel_error = vec3d_norm(error_x, error_y, error_z) / ref_norm
                print("Relative error in {}: {}".format(force_moment_test_type, rel_error))

                if rel_error > tol:
                    raise ValueError("ERROR: Relative error is {}, which is greater " \
                                    "than the tolerance {}".format(rel_error, tol))
        assert force_moment_test_type_found, "Must set --force_moment_test_type"
    elif solve_type == "calculate_weight_data":
        force_moment_test_type_found = False
        for output_data in fe_response.output_datas:
            if output_data.output_type == force_moment_test_type:
                force_moment_test_type_found = True
                elements = []

                if force_moment_test_type == "Weight":
                    units = "mass"
                    elements.append(output_data.real_valued_data["weight"])
                elif force_moment_test_type == "Center of gravity":
                    units = "mesh_length"
                    elements.append(output_data.real_valued_data["element_x"])
                    elements.append(output_data.real_valued_data["element_y"])
                    elements.append(output_data.real_valued_data["element_z"])
                elif force_moment_test_type == "I_ij":
                    units = "mass_moment_of_inertia_metric"
                    elements.append(output_data.real_valued_data["I_xx"])
                    elements.append(output_data.real_valued_data["I_yy"])
                    elements.append(output_data.real_valued_data["I_zz"])
                    elements.append(output_data.real_valued_data["I_xy"])
                    elements.append(output_data.real_valued_data["I_xz"])
                    elements.append(output_data.real_valued_data["I_yz"])
                else:
                    assert False, "Invalid force_moment_test_type"

                _, encoded_elements, units_str = dv.encode_params(
                    collection, "", elements, units)

                if force_moment_test_type == "Weight":
                    computed_weight = encoded_elements[0]

                    print("Computed weight: {} {}".format(
                        computed_weight, units_str))

                    rel_error = abs(computed_weight - ref_x) / ref_x
                elif force_moment_test_type == "Center of gravity":
                    computed_x = encoded_elements[0]
                    computed_y = encoded_elements[1]
                    computed_z = encoded_elements[2]

                    print("Computed CoG: ({},{},{}) {}".format(
                        computed_x, computed_y, computed_z, units_str))

                    error_x = abs(computed_x - ref_x)
                    error_y = abs(computed_y - ref_y)
                    error_z = abs(computed_z - ref_z)
                    rel_error = vec3d_norm(error_x, error_y, error_z) / vec3d_norm(ref_x, ref_y, ref_z)
                elif force_moment_test_type == "I_ij":
                    computed_I_xx = encoded_elements[0]
                    computed_I_yy = encoded_elements[1]
                    computed_I_zz = encoded_elements[2]
                    computed_I_xy = encoded_elements[3]
                    computed_I_xz = encoded_elements[4]
                    computed_I_yz = encoded_elements[5]

                    print("Computed moments of inertia: ({},{},{},{},{},{}) {}".format(
                        computed_I_xx, computed_I_yy, computed_I_zz, computed_I_xy, computed_I_xz, computed_I_yz, units_str))

                    error_x = abs(computed_I_xx - ref_x)
                    error_y = abs(computed_I_yy - ref_y)
                    error_z = abs(computed_I_zz - ref_z)
                    error_xy = abs(computed_I_xy - ref_xy)
                    error_xz = abs(computed_I_xz - ref_xz)
                    error_yz = abs(computed_I_yz - ref_yz)

                    error_norm = math.sqrt(error_x**2 + error_y**2 + error_z**2 + error_xy**2 + error_xz**2 + error_yz**2)
                    ref_norm = math.sqrt(ref_x**2 + ref_y**2 + ref_z**2 + ref_xy**2 + ref_xz**2 + ref_yz**2)
                    rel_error = error_norm / ref_norm
                else:
                    assert False, "Invalid force_moment_test_type"

                print("Relative error in {}: {}".format(force_moment_test_type, rel_error))

                if rel_error > tol:
                    raise ValueError("ERROR: Relative error is {}, which is greater " \
                                    "than the tolerance {}".format(rel_error, tol))
        assert force_moment_test_type_found, "Must set --force_moment_test_type"
    else:
        if solve_type == "plot_elasticity_materials" or solve_type == "plot_thermoelasticity_materials":
            assert compare_component_solutions, "Must use --compare-component-solutions"

        if not compare_component_solutions:
            solution_exo_partial_path = [f for f in fe_response.files
                                        if os.path.basename(f).startswith("global_solution")][0]

            full_reference_solution_exo_filename = os.path.join(
                directories.DATA_DIR, "reference_data", reference_solution_exo_filename)
            print("Reference solution exo filename (first): ", full_reference_solution_exo_filename)

            solution_exo_filename = os.path.join(
                directories.OUTPUT_FILES_DIRECTORY, solution_exo_partial_path)
            print("Solution exo filename (second): ", solution_exo_filename)

            if not do_not_compare_exo_files:
              compare_exo_files(full_reference_solution_exo_filename, solution_exo_filename, tol)
        else:
            component_solution_prefix = "component_solution"
            component_solution_exo_partial_paths = \
                sorted([f for f in fe_response.files if os.path.basename(f).startswith(component_solution_prefix)])

            for component_solution_exo_partial_path in component_solution_exo_partial_paths:
                # get the component solution suffix from the component solution filename
                component_solution_suffix = os.path.basename(component_solution_exo_partial_path)[len(component_solution_prefix):]
                # to get the component reference solution exo filename, we also remove ".gz" or ".bz2" from the suffix
                if component_solution_suffix.endswith( ".gz" ):
                    component_solution_suffix = component_solution_suffix[:-3]
                elif component_solution_suffix.endswith( ".bz2" ):
                    component_solution_suffix = component_solution_suffix[:-4]

                component_reference_solution_exo_filename = reference_solution_exo_filename[:-4] + component_solution_suffix
                full_component_reference_solution_exo_filename = os.path.join(
                    directories.DATA_DIR, "reference_data", component_reference_solution_exo_filename)
                print("Component reference solution exo filename(first): ", full_component_reference_solution_exo_filename)

                component_solution_exo_filename = os.path.join(
                    directories.OUTPUT_FILES_DIRECTORY, component_solution_exo_partial_path)
                print("Component solution exo filename (second): ", component_solution_exo_filename)

                if not do_not_compare_exo_files:
                    compare_exo_files(full_component_reference_solution_exo_filename, component_solution_exo_filename, tol)


if __name__ == "__main__":
    print("START FE SOLVER STABILITY TEST")
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("aks_filename")
    # parser.add_argument("reference_solution_exo_filename")
    parser.add_argument("--mesh-stitching-tolerance", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--num-cores", type=int, default=1)
    parser.add_argument("--use-elem-stress", default=False, action='store_true')
    # The --von-mises-stress-only flag skips writing stress/strain components and instead
    # just computes and writes the Von Mises stress, which can be useful for testing
    # stresses without writing quite so much data.
    parser.add_argument("--von-mises-stress-only", default=False, action='store_true')
    parser.add_argument("--use-global-coords-for-surface-constraints", default=False, action='store_true')
    parser.add_argument("--is-shell", default=False, action='store_true')
    parser.add_argument("--do-not-compare-exo-files", default=False, action='store_true')
    parser.add_argument("--solve-type", type=str, default='default')
    parser.add_argument("--contact-algorithm", type=str, default='Augmented Lagrangian')
    parser.add_argument("--finite-sliding", action='store_true', default=False)
    parser.add_argument("--uncoupled-submodel-solve", default=False, action='store_true')
    parser.add_argument("--submodel-ids", nargs='+', type=int)
    parser.add_argument("--time-parameters", nargs='+', type=float)
    parser.add_argument("--max-dofs-per-port", type=int, default=-1)
    parser.add_argument("--allow-nonconforming-port-meshes", action='store_true', default=False)
    parser.add_argument("--use-reduced-integration", action='store_true', default=False)
    parser.add_argument("--use-parallel-mesh-stitching", action='store_true', default=False)
    parser.add_argument("--ref_x", type=float)
    parser.add_argument("--ref_y", type=float)
    parser.add_argument("--ref_z", type=float)
    parser.add_argument("--ref_xy", type=float, default=0.0)
    parser.add_argument("--ref_xz", type=float, default=0.0)
    parser.add_argument("--ref_yz", type=float, default=0.0)
    parser.add_argument("--origin_x", type=float, default=0.0)
    parser.add_argument("--origin_y", type=float, default=0.0)
    parser.add_argument("--origin_z", type=float, default=0.0)
    parser.add_argument("--force_moment_test_type", type=str)
    parser.add_argument("--ref_units", type=str)
    parser.add_argument("--skip-precomputed-interface-functions", action='store_true', default=False)
    parser.add_argument("--include-materials-in-plot", action='store_true', default=False)
    parser.add_argument("--compare-component-solutions", action='store_true', default=False)
    parser.add_argument("--do-not-use-component-datasets", action='store_true', default=False)

    args = parser.parse_args()
    args.aks_filename = "hemisphere.aks"
    args.reference_solution_exo_filename = "hemisphere.exo"
    args.is_shell = True

    # test_fe_solver(os.path.abspath("aks_files/" + args.aks_filename),
    test_fe_solver(os.path.abspath("aks_files/" + args.aks_filename),
                   args.reference_solution_exo_filename,
                   args.num_cores,
                   args.mesh_stitching_tolerance,
                   args.tolerance,
                   args.use_elem_stress,
                   args.von_mises_stress_only,
                   args.use_global_coords_for_surface_constraints,
                   args.is_shell,
                   args.do_not_compare_exo_files,
                   args.solve_type,
                   args.uncoupled_submodel_solve,
                   args.submodel_ids,
                   args.time_parameters,
                   args.contact_algorithm,
                   args.finite_sliding,
                   args.ref_x, args.ref_y, args.ref_z,
                   args.ref_xy, args.ref_xz, args.ref_yz,
                   args.origin_x, args.origin_y, args.origin_z,
                   args.force_moment_test_type, args.ref_units,
                   args.skip_precomputed_interface_functions,
                   args.include_materials_in_plot,
                   args.max_dofs_per_port,
                   allow_nonconforming_port_meshes=args.allow_nonconforming_port_meshes,
                   use_reduced_integration=args.use_reduced_integration,
                   use_parallel_mesh_stitching=args.use_parallel_mesh_stitching,
                   compare_component_solutions=args.compare_component_solutions,
                   do_not_use_component_datasets=args.do_not_use_component_datasets)
    print("FINISH !")
