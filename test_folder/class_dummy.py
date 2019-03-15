
class Employee:
    numb = 0
    raiseamount = 1.04
    def __init__(self, first, last, pay):
        self.first = first
        self.last =  last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        Employee.numb += 1 #constant này không cần thay đổi nên sẽ là Employee.

    def fullname(self):
        return  '{} {}'.format(self.first,self.last)

    def apply_raise(self):
        self.pay = int(self.pay*self.raiseamount) #constant raiseamount sẽ cần thay đổi nên sẽ là self.raiseamount

    @classmethod
    def set_raise_amt(cls, amount):
        cls.raiseamount = amount

    @classmethod
    def from_string(cls,staff_str):
        first, last,pay = staff_str.split('-')
        return cls(first,last,pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 5 :
            return False
        return True

class Developer(Employee):
    pass

dev1 = Developer('Son','Le',12000)
dev2 = Developer('Son','Duong',15000)
print(dev1.email)



# staff_str_3 = 'David-Coper-70000'
# staff_str_4 = 'Lu-La-60000'
# staff_str_5 = 'Ga-Chien-40000'

# staff3 = Employee.from_string(staff_str_3)
# staff4 = Employee.from_string(staff_str_4)
# staff5 = Employee.from_string(staff_str_5)
#
# print(staff3.email)
