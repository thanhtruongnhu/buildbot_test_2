# Copyright (C) 2015 Akselos
import os
import sys
import tempfile
import types
import math

import numpy as np
import netCDF4

import set_akselos_path  # noqa
import directories
import akselos.utils
import types_data
import json_helper
import server.run_exe
import interfaces.akselos_assembly
import interfaces.scrbe_solver_args
import interfaces.component_visualization_args
import interfaces.rb_solution
import train.job_options as jo
import scipy.spatial as sp
import quick_load_asl as ql


def complex_from_string(string):
    try:
        return complex(string)
    except ValueError:
        # The 'value' attribute is a string in the form "(-1.2392e-3,1.98765e-4)"
        # (we trust the RB solver to not send us harmful Python code here!)
        return complex(*eval(string))


def compare(object1, object2, tol, max_abs=None, exo_coords_1=None, exo_coords_2=None,
            check_linear_dependence=False):
    # -------------------
    # This helper function compares basic types and takes care of
    # correctly comparing floats
    def assert_equal(arg1, arg2):
        try:
            float(arg1)
            float(arg2)
            arg1 = float(arg1)
            arg2 = float(arg2)
        except ValueError:
            pass
        if isinstance(arg1, float):
            try:
                assert abs(arg1-arg2)/arg1 < tol, \
                    "Relative difference between {} and {} is > {}".format(arg1, arg2, tol)
            except ZeroDivisionError:
                assert not arg1 == 0.0 != arg2 == 0.0, \
                    "No equality between {} and {}".format(arg1, arg2)
        else:
            assert arg1 == arg2, "No equality between\n {}\n and\n {}\n".format(arg1, arg2)
    # -------------------

    # First check that the elements have the same type
    assert type(object1) is type(object2), "{} and {} have different types".format(object1, object2)

    # Case of a Numpy array
    if isinstance(object1, np.ndarray):
        if max_abs is None:
            norm = max(np.max(np.abs(object1)), np.max(np.abs(object2)))
        else:
            norm = max_abs
        atol = tol*norm+1e-8
        if not np.allclose(object1, object2, rtol=0.0, atol=atol):
            flat1 = np.array(list(object1.flat))
            flat2 = np.array(list(object2.flat))
            args_sorted = list(reversed(np.argsort(abs(flat1 - flat2))))
            print("Arrays are not equal, atol:", atol, "max", norm, file=sys.stderr)
            print("Ten biggest discrepancies:", file=sys.stderr)
            print("| node idx | node 1 value | node 2 value | node 1 position | node 2 position |")
            for idx, o1, o2, p1, p2 in zip(args_sorted[:10],
                flat1[args_sorted][:10], flat2[args_sorted][:10],
                exo_coords_1[args_sorted][:10], exo_coords_2[args_sorted][:10]):
                print(idx, o1, o2, p1, p2, file=sys.stderr)
            raise ValueError("arrays are not equal within tolerance")

    elif isinstance(object1, netCDF4.Dataset):
        compare(list(object1.variables.keys()), list(object2.variables.keys()), tol)
        if 'name_nod_var' in object1.variables and 'name_nod_var' in object2.variables:
            max_abs1s = get_max_abs_displacement(object1)
            max_abs2s = get_max_abs_displacement(object2)
            coords_1 = get_exo_coords(object1)
            coords_2 = get_exo_coords(object2)
            node_field_names = [x.tostring().split(b'\x00')[0].decode('utf-8') for x in object1.variables['name_nod_var']]
            assert len(coords_1) == len(coords_2), \
                "Number of nodes in two EXO files are not equal: " + str(len(coords_1)) + " vs. " +\
                str(len(coords_2))

            identical_node_orders = np.allclose(coords_1, coords_2)
            if identical_node_orders:
                if check_linear_dependence:
                    compare_linear_dependence(
                        node_field_names, object1, object2, tol)
                else:
                    compare_nodal_values(
                        node_field_names, object1, object2, max_abs1s, max_abs2s, tol,
                        coords_1, coords_2)

            else:
                print("The two EXO files have different node orders. Mapping them...")
                kdtree = sp.KDTree(coords_1)
                distances, indices = kdtree.query(coords_2)

                is_all_close = np.allclose(
                    distances, np.zeros(len(distances)), rtol=1e-4, atol=1e-4)
                if not is_all_close:
                    max_distance = max(distances)
                    max_idx = np.argmax(distances)
                    print("Largest node distance =", max_distance)
                    print("   between node", indices[max_idx], "of EXO 1:", \
                        coords_1[indices[max_idx]])
                    print("   and node", max_idx, "of EXO 2:", coords_2[max_idx])
                    assert False, 'Could not map 2 sets of nodes within tolerance.'

                unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
                if len(unique_indices) < len(coords_2):
                    print('Kdtree could not find one-to-one mapping, perhaps because EXO files ' \
                          'have multiple nodes at one position (i.e. contact model). Try to find ' \
                          'better mapping...')

                    non_unique_mapping_from_1_to_2 = {}
                    for i in range(len(inverse_indices)):
                        assert (inverse_indices == inverse_indices[i]).sum() > 0
                        if (inverse_indices == inverse_indices[i]).sum() > 1:
                            if indices[i] not in non_unique_mapping_from_1_to_2.keys():
                                non_unique_mapping_from_1_to_2[indices[i]] = [i]
                            else:
                                non_unique_mapping_from_1_to_2[indices[i]].append(i)

                    missing_indices = []
                    for i in range(len(coords_1)):
                        if i not in indices:
                            missing_indices.append(i)

                    distances_k2, indices_k2 = kdtree.query(coords_2, k=2)
                    for exo_1_node_idx in non_unique_mapping_from_1_to_2:
                        exo_2_node_idxs = non_unique_mapping_from_1_to_2[exo_1_node_idx]

                        # Check error from all fields between reference node in EXo2 and first
                        # target node in EXO 1
                        test_error_1 = check_error_from_all_fields(
                            object1, object2, exo_1_node_idx, exo_2_node_idxs[0],
                            len(node_field_names))

                        # Check error from all fields between reference node in EXo2 and second
                        # target node in EXO 1
                        test_error_2 = check_error_from_all_fields(
                            object1, object2, exo_1_node_idx, exo_2_node_idxs[1],
                            len(node_field_names))

                        # Re-mapping the one has larger error
                        if test_error_1 < test_error_2:
                            wrong_node_idx = 1
                        else:
                            wrong_node_idx = 0
                        wrong_node_idx_in_exo_2 = exo_2_node_idxs[wrong_node_idx]
                        # Note that the matching node from kdtree.query(coords_2, k=1) might not be
                        # the first matching from kdtree.query(coords_2, k=2). It can be the second
                        # because actually the first and second are identical in term of distance.
                        other_match_in_exo_1 = \
                            set(indices_k2[wrong_node_idx_in_exo_2]) - set([exo_1_node_idx])
                        other_match_in_exo_1 = list(other_match_in_exo_1)[0]
                        assert other_match_in_exo_1 in missing_indices, 'Error: node ' \
                            'already in used. Could not assign it.'
                        print("Node {} in EXO 2 is now mapped to node {} in EXO 1, " \
                              "before was node {}.".format(wrong_node_idx_in_exo_2,
                              other_match_in_exo_1, indices[wrong_node_idx_in_exo_2]))
                        indices[wrong_node_idx_in_exo_2] = other_match_in_exo_1
                        #distances[wrong_node_idx_in_exo_2] = distances_k2[wrong_node_idx_in_exo_2][kx]

                    unique_indices, _ = np.unique(indices, return_inverse=True)
                    assert len(unique_indices) == len(coords_2), \
                        "Error: Mapping is still not one-to-one after updated."

                    is_all_close = np.allclose(
                        distances, np.zeros(len(distances)), rtol=1e-4, atol=1e-4)
                    if not is_all_close:
                        max_distance = max(distances)
                        max_idx = np.argmax(distances)
                        print("Largest node distance =", max_distance)
                        print("   between node", indices[max_idx], "of EXO 1:", \
                            coords_1[indices[max_idx]])
                        print("   and node", max_idx, "of EXO 2:", coords_2[max_idx])
                        assert False, 'Could not map 2 sets of nodes within tolerance (2).'

                # Enable this to print node mapping details:
                #coords_1_idxs = np.arange(len(coords_1))
                #diff = indices - coords_1_idxs
                #diff_idxs = np.where(diff != 0)[0]
                #print "Found node mapping (from EXO 1 nodes to EXO 2 nodes):"
                #for diff_idx in diff_idxs:
                #    print ' ', coords_1_idxs[diff_idx], '->', indices[diff_idx]
                if check_linear_dependence:
                    compare_linear_dependence(
                        node_field_names, object1, object2, tol, mapping_indices=indices)
                else:
                    compare_nodal_values(
                        node_field_names, object1, object2, max_abs1s, max_abs2s, tol,
                        coords_1, coords_2, mapping_indices=indices)

        else:
            print("Skip checking max displacements because the exo file(s) don't have " \
                  "'name_nod_var' variable.")

        # Check element fields values if they exist
        if 'name_elem_var' in object1.variables and 'name_elem_var' in object2.variables:
            print("> Checking element fields:")
            field_values_1 = get_element_field_values(object1)
            field_values_2 = get_element_field_values(object2)
            assert len(field_values_1) == len(field_values_2) # number of blocks should be equal
            for block_idx in range(len(field_values_1)):
                print(" + Block idx", block_idx)
                block_values_1 = field_values_1[block_idx]
                block_values_2 = field_values_2[block_idx]
                for key in block_values_1:
                    assert key in block_values_2, "Could not find " + str(key) \
                                                  + " field in second exo."
                    values_1 = block_values_1[key]
                    values_2 = block_values_2[key]
                    normalization = np.max(np.abs(values_2))
                    abs_tol = max(tol*normalization, 1.e-10)
                    matched = np.allclose(values_1, values_2, rtol=0.0, atol=abs_tol)
                    assert matched, \
                        "not matched at field '" + key + "' of block idx " + str(block_idx)

    # Case of an iterable
    elif isinstance(object1, (list, tuple)):
        if not len(object1) == len(object2):
            print("Difference between object1 and object2:")
            print(object1)
            print(object2)
            assert False
        
        for element1, element2 in zip(object1, object2):
            compare(element1, element2, tol)

    # compare set, assume stable sort
    elif isinstance(object1, set):
        assert len(object1) == len(object2)
        for element1, element2 in zip(sorted(list(object1)), sorted(list(object2))):
            compare(element1, element2, tol)

    # Case of a standard type
    elif isinstance(object1, (str, bytes, bool, int, float, complex)):
        assert_equal(object1, object2)

    # Case of a class instance (default in python3 new-style class)
    else:

        # Check that they're instances of the same class
        assert object1.__class__ == object2.__class__, \
            "{} and {} are instances of different classes".format(object1.__class__,
                                                                  object2.__class__)

        # Check that they have the same attributes
        assert_equal(list(object1.__dict__.keys()), list(object2.__dict__.keys()))

        # Compare each attribute
        for attribute in object1.__dict__:

            # Check that they both have the attribute
            assert not hasattr(object1, attribute) != hasattr(object2, attribute), \
                "{} and {} don't both have attribute {}".format(object1, object2, attribute)

            # Special treatment of the RB coefficient list
            if attribute == 'coefficients' and isinstance(object1,
                                                          interfaces.rb_solution.RBSolution):
                l2_norm = 0.0

                # Compute the L2 norm of the coefficient vector
                for coef1, coef2 in zip(getattr(object1, attribute), getattr(object1, attribute)):
                    # Check that both coefficients have the same attributes and indices
                    compare(list(coef1.__dict__.keys()), list(coef2.__dict__.keys()), tol)
                    for attr in ['dof_index', 'value']:
                        assert hasattr(coef1, attr), "{} doesn't have attribute {}" \
                            .format(coef1, attr)
                        assert hasattr(coef2, attr), "{} doesn't have attribute {}" \
                            .format(coef2, attr)
                    assert_equal(getattr(coef1, attr), getattr(coef2, attr))

                    # The 'value' attribute is a string in the form "(-1.2392e-3,1.98765e-4)"
                    # (we trust the RB solver to not send us harmful Python code here!)
                    l2_norm += abs(complex_from_string(coef1.value)) ** 2
                l2_norm = math.sqrt(l2_norm)

                # Check each coefficient
                for coef1, coef2 in zip(getattr(object1, attribute), getattr(object1, attribute)):
                    err = abs(complex_from_string(coef1.value) - complex_from_string(coef2.value)) \
                          / l2_norm
                    if err >= tol:
                        message = "Coefficients {} and {} have too great of an error " \
                                  "(relative to the L2 norm of the coefficient vector)."
                        raise ValueError(message.format(coef1.value, coef2.value))

            else:
                compare(getattr(object1, attribute), getattr(object2, attribute), tol)


def decompress_file(filepath):
    _, filename = os.path.split(filepath)
    with open(filepath, 'rb') as f:
        data = f.read()

    data, filename = akselos.utils.decompress_data(data, filename)

    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(data)
    fname = tmp_file.name
    tmp_file.close()
    return fname


def compare_exo_files(exo1, exo2, tol, check_linear_dependence=False):
    use_exo1 = use_exo2 = None
    try:
        use_exo1 = decompress_file(exo1)
        use_exo2 = decompress_file(exo2)
        loaded1 = netCDF4.Dataset(use_exo1)
        loaded2 = netCDF4.Dataset(use_exo2)
        compare(loaded1, loaded2, tol, check_linear_dependence=check_linear_dependence)
    finally:
        if use_exo1 is not None and use_exo1 != exo1:
            os.unlink(use_exo1)

        if use_exo2 is not None and use_exo2 != exo2:
            os.unlink(use_exo2)


def check_error_from_all_fields(f1, f2, node_1_idx, node_2_idx, n_node_field_names):
    errors = np.zeros(n_node_field_names)
    for i in range(n_node_field_names):
        v1 = f1.variables['vals_nod_var'+str(i+1)][:][0][node_1_idx]
        v2 = f2.variables['vals_nod_var'+str(i+1)][:][0][node_2_idx]
        errors[i] = v1 - v2
    return np.linalg.norm(errors)


def get_exo_coords(f):
    if 'coord' in f.variables:
        coords = f.variables['coord'][:].transpose()
    else:
        x_coords = f.variables['coordx'][:]
        y_coords = f.variables['coordy'][:]
        z_coords = f.variables['coordz'][:]
        coords = np.vstack((x_coords, y_coords, z_coords)).T
    return coords


def get_max_abs_displacement(dataset):
    # If we have arrays with x, y, z components, we want the tolerance to be relative to the
    # maximum total length, not one separate maximum per component.
    # Currently, we recognize "u", "v", "w" and "theta_x", "theta_y", "theta_z"
    # arrays from an exo-specific netCDF4 dataset.
    strings_array = dataset.variables['name_nod_var'][:]
    strings = [x.tostring().split(b'\x00')[0].decode('utf-8') for x in strings_array]
    keys = list(dataset.variables.keys())
    trios = [('u', 'v', 'w'),
             ('theta_x', 'theta_y', 'theta_z')]
    result = {}
    for trio in trios:
        if not all(trio[i] in strings for i in range(3)):
            continue

        xyzkeys = ["vals_nod_var" + str(strings.index(trio[i]) + 1) for i in range(3)]
        if not all(xyzkeys[i] in keys for i in range(3)):
            continue

        v0, v1, v2 = [dataset.variables[xyzkeys[i]][:] for i in range(3)]
        mx = np.max(np.sqrt(v0*v0 + v1*v1 + v2*v2))
        for i in range(3):
            result[xyzkeys[i]] = mx
    return result


def get_element_field_values(dataset):
    if 'name_elem_var' in dataset.variables:
        strings_array = dataset.variables['name_elem_var'][:]
        elem_field_names = [x.tostring().split(b'\x00')[0].decode('utf-8') for x in strings_array]
        result = []
        for block_idx in range(len(dataset.variables["eb_prop1"][:])):
            block_result = {}
            for idx, elem_field_name in enumerate(elem_field_names):
                key = "vals_elem_var" + str(idx+1) + "eb" + str(block_idx+1)
                if key not in dataset.variables:
                    continue
                block_result[elem_field_name] = dataset.variables[key][:]
                result.append(block_result)
        return result


def compare_nodal_values(node_field_names, object1, object2, max_abs1s, max_abs2s, tol, coords_1,
                         coords_2, mapping_indices=None):
    if mapping_indices is not None:
        coords_1 = coords_1[mapping_indices]
    for key_idx in range(len(node_field_names)):
        key = 'vals_nod_var' + str(key_idx+1)
        max_abs1 = max_abs1s.get(key, None)
        max_abs2 = max_abs2s.get(key, None)
        use_max_abs = None
        if max_abs1 is not None and max_abs2 is not None:
            use_max_abs = max(max_abs1, max_abs2)
        print("> Check node field:", node_field_names[key_idx])
        values_1 = object1.variables[key][:]
        values_2 = object2.variables[key][:]
        if mapping_indices is not None:
            values_1 = values_1[:, mapping_indices]
        compare(values_1, values_2, tol, use_max_abs,
                exo_coords_1=coords_1, exo_coords_2=coords_2)

def compute_linear_dependence(values_1,values_2):

    assert np.shape(values_1) == np.shape(values_2)

    norm2_values_1 = np.dot(values_1[:],values_1[:])
    norm2_values_2 = np.dot(values_2[:],values_2[:])

    dot2_values_1_values_2 = np.square(np.dot(values_1,values_2))

    return dot2_values_1_values_2 / ( norm2_values_1 * norm2_values_2 )


def compare_linear_dependence(node_field_names, object1, object2, tol, mapping_indices=None):

    all_values_1 = np.zeros((1,0))
    all_values_2 = np.zeros((1,0))
    for key_idx in range(len(node_field_names)):
        key = 'vals_nod_var' + str(key_idx+1)
        values_1 = object1.variables[key][:]
        values_2 = object2.variables[key][:]
        if mapping_indices is not None:
            values_1 = values_1[:, mapping_indices]
        all_values_1 = np.concatenate((all_values_1, values_1),axis=1)
        all_values_2 = np.concatenate((all_values_2, values_2),axis=1)

    squared_cosine = compute_linear_dependence(all_values_1[0,:], all_values_2[0,:])
    print("linear dependence =", squared_cosine)

    # If values_1 and values_2 are linearly dependent, the squared cosine is 1
    if abs(squared_cosine-1) > tol:
        raise ValueError("Arrays are not linearly dependent within tolerance")


def test_rb_solver_stability(aks_file, print_answer=None, scrbe_mpi_np=1, max_dofs_per_port=-1,
                             n_eigenvalues=1, compare_component_viz=False, ref_solution_prefix=None,
                             tolerance=1e-6, use_stress=False, use_fe_solves_for_viz=True):

    types = types_data.TypesData()
    filename = aks_file

    akselos_assembly = json_helper.read(interfaces.akselos_assembly.AkselosAssembly,
                                        open(filename).read())
    solver_parameters = interfaces.component_system_data.SolverOptionsData()
    solver_parameters.solver_strategy = "rb"
    solver_parameters.solver_type = "rb_default_elasticity"
    solver_parameters.num_cores = scrbe_mpi_np

    collection_name = akselos_assembly.component_system.collection_type
    physics_name = types.get_collection(collection_name).get_physics_name().upper()

    is_eigen_collection = physics_name == "ELASTICITY_EIGEN"

    if max_dofs_per_port >= 0:
        solver_parameters.int_valued_data_present = True
        solver_parameters.int_valued_data["max_dofs_per_port"] = max_dofs_per_port

    modified_scrbe_mpi_np = scrbe_mpi_np
    n_solve_requests = 1
    if is_eigen_collection:
        n_solve_requests = n_eigenvalues
        solver_parameters.solver_type = "rb_elasticity_eigen"
        solver_parameters.int_valued_data_present = True
        solver_parameters.int_valued_data["Total number of eigenvalues"] = n_eigenvalues
        modified_scrbe_mpi_np = 1

    akselos_assembly.component_system.solver_options_data_present = True
    akselos_assembly.component_system.solver_options_data = [solver_parameters]
    for solve_request_idx in range(n_solve_requests):
        if is_eigen_collection:
            # Set the eigenvalue index we solve for
            eigenvalue_idx = solve_request_idx + 1
            solver_parameters.int_valued_data["Number of eigenvalues"] = eigenvalue_idx
            print("\n# Eigenvalue index =", eigenvalue_idx)

        scrbe_args = interfaces.scrbe_solver_args.ScrbeSolverSolve(
            component_system=akselos_assembly.component_system)

        print("Solving", os.path.basename(aks_file))
        # scrbe_response_text, = server.run_exe.run_scrbe_process_text(scrbe_args, mpi_np=scrbe_mpi_np)
        # scrbe_response = json_helper.read(interfaces.scrbe_solver_args.ScrbeSolverResponse,
        #                                   scrbe_response_text)
        job_options = jo.JobOptions(ppn=modified_scrbe_mpi_np)
        scrbe_response = server.run_exe.run_scrbe_process(scrbe_args, job_options)

        if print_answer is None:
            pass
            """ This blocks was used to compare the RB solution vector against a reference
            with open('tests.json') as f:
                json_data = json.load(f)
                fname = os.path.basename(__file__)
                reference_file_name = json_data[fname]['targets'][basename]['reference_solution']
                test_case_folder = json_data[fname]['folder']
                reference_file_path = os.path.join(test_case_folder, reference_file_name)


            with open(reference_file_path, 'r') as f:
                reference_data = f.read()
            """

        print("\nResponse from RB solver succesfully parsed.")

        if compare_component_viz:
            if ref_solution_prefix is None:
                assert False, "No reference solution provided"

            for component_idx, rb_solution in enumerate(scrbe_response.rb_solutions):
                # Skip "built in" components
                if rb_solution.ref_component_name.startswith("/builtin"):
                    continue

                print("\n - Visualize solution for component_idx = {}: {} ...".format(component_idx, rb_solution.ref_component_name))

                viz_parameters = interfaces.visualization_parameters.VisualizationParameters()
                viz_parameters.boolean_data_present = True
                viz_parameters.boolean_data["plot_exodus"] = True
                viz_parameters.boolean_data["no_viz_cache"] = True
                viz_parameters.boolean_data["single_component_fe_solve_for_viz"] = use_fe_solves_for_viz

                if use_stress:
                    viz_parameters.extra_fields_present = True
                    viz_parameters.extra_fields.append("stress_tensor")
                    viz_parameters.boolean_data["use_element_stress"] = True

                # In case of FE viz, we also need to provide the eigenvalue for the viz process,
                # so we just add it regardless of the viz type.
                if is_eigen_collection:
                    eigenvalue = scrbe_response.eigenvalue
                    rb_solution.real_valued_data_present = True
                    rb_solution.real_valued_data["eigenvalue"] = eigenvalue

                plot_args = interfaces.component_visualization_args.ComponentVisualizationPlot(
                    visualization_parameters=viz_parameters,
                    compute_l2_error_id=None,
                    rb_component_solution=rb_solution)

                try:
                    viz_response = server.run_exe.run_scrbe_process(plot_args)
                except ValueError:
                    print("A JSON decoding error occurred during the viz.")
                    sys.exit(1)

                viz_file = os.path.join(directories.OUTPUT_FILES_DIRECTORY, viz_response.result_filepath)

                if viz_file.endswith('.bz2') or viz_file.endswith('.gz'):
                    viz_file = decompress_file(viz_file)

                ref_solution_exo_filename = ref_solution_prefix
                if is_eigen_collection:
                    ref_solution_exo_filename += "_mode_" + str(eigenvalue_idx)
                    ref_solution_exo_filename += "_component_" + str(component_idx) + ".exo"
                full_ref_solution_exo_filename = os.path.join(
                    directories.DATA_DIR, "reference_data", ref_solution_exo_filename)

                print("Comparing to component reference solution file", ref_solution_exo_filename)

                compare_exo_files(viz_file, full_ref_solution_exo_filename, tolerance,
                                  check_linear_dependence=is_eigen_collection)


    """ This blocks was used to compare the RB solution vector against a reference
    if print_answer is None:
        try:
            scrbe_reference_response = json_helper.read(scrbe_response.__class__, reference_data)
        except ValueError:
            print "Could not decode the reference RB response:"
            print reference_data
            sys.exit(1)
        
        try:
            # Ignore the timings, which are obviously different 
            scrbe_response.timings = None
            scrbe_reference_response.timings = None
            
            compare(scrbe_response, scrbe_reference_response, tol)
            
        except AssertionError as e:
            print "\nComparison failed for model {}:".format(os.path.basename(aks_file))
            log_file = aks_file + ".faulty_response"
            with open(log_file, 'w') as f:
                f.write(scrbe_response_text)
            print e
            print "Dumping faulty response to {}".format(os.path.basename(log_file))
            print "Run the following on PC1 to see the difference (reference is on the left, and timings are ignored in any case):"
            print "diff {} {} | diff | less" \
                .format(os.path.abspath(reference_file_path), os.path.abspath(log_file))
            sys.exit(1)
              
        print "\nThe response matches the reference."
        
    else:
        with open(print_answer, 'w') as f:
            f.write(scrbe_response_text)
            print "Response from RB solver written to {}".format(print_answer)
    """


def compare_asls(asl_0, asl_1):
    print("> Comparing two .asl files")
    print(asl_0)
    print(asl_1)
    solution_0 = ql.quick_load_asl(asl_0)
    solution_1 = ql.quick_load_asl(asl_1)
    tolerance = 1e-4

    assert len(solution_0.component_solutions) == len(solution_1.component_solutions)
    for comp_idx in range(len(solution_0.component_solutions)):
        print(" Comparing component idx", comp_idx)
        solution_0_comp_viz = solution_0.component_solutions[comp_idx].fb_solution
        solution_1_comp_viz = solution_1.component_solutions[comp_idx].fb_solution
        exo_coords_0 = solution_0_comp_viz.ref_mesh.mesh_data.coords
        exo_coords_1 = solution_1_comp_viz.ref_mesh.mesh_data.coords

        compare(solution_0_comp_viz.node,
                solution_1_comp_viz.node,
                tolerance)
        compare(sorted(list(solution_0_comp_viz.ufields.keys())),
                sorted(list(solution_1_comp_viz.ufields.keys())),
                tolerance)
        for uname in solution_0_comp_viz.ufields:
            print('    comparing field', uname)
            compare(solution_0_comp_viz.ufields[uname],
                    solution_1_comp_viz.ufields[uname],
                    tolerance,
                    exo_coords_1=exo_coords_0,
                    exo_coords_2=exo_coords_1)

        solution_0_comp_elem_fields = solution_0_comp_viz.get_elem_field_names()
        solution_1_comp_elem_fields = solution_1_comp_viz.get_elem_field_names()
        compare(solution_0_comp_elem_fields,
                solution_1_comp_elem_fields,
                tolerance)
        for ename in solution_0_comp_elem_fields:
            efield_0 = solution_0_comp_viz.get_all_elem_field_values(ename)
            efield_1 = solution_1_comp_viz.get_all_elem_field_values(ename)

            if len(efield_0) == 0 and len(efield_1)==0:
                continue
            print('   comparing field', ename)
            compare(efield_0, efield_1, tolerance, exo_coords_1=exo_coords_0,
                    exo_coords_2=exo_coords_1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("aks_filename")
    parser.add_argument("--scrbe-mpi-np", type=int, default=1)
    parser.add_argument("--max-dofs-per-port", type=int, default=-1)
    parser.add_argument("--n-eigenvalues", type=int, default=1)
    parser.add_argument("--compare-component-viz", action='store_true', default=False)
    parser.add_argument("--ref-solution-prefix", type=str, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--use-stress", action='store_true', default=False)
    parser.add_argument("--use-fe-solves-for-viz", action='store_true', default=False)

    args = parser.parse_args()

    test_rb_solver_stability(args.aks_filename,
                             scrbe_mpi_np=args.scrbe_mpi_np,
                             max_dofs_per_port=args.max_dofs_per_port,
                             n_eigenvalues=args.n_eigenvalues,
                             compare_component_viz=args.compare_component_viz,
                             ref_solution_prefix=args.ref_solution_prefix,
                             tolerance=args.tolerance,
                             use_stress=args.use_stress,
                             use_fe_solves_for_viz=args.use_fe_solves_for_viz)
