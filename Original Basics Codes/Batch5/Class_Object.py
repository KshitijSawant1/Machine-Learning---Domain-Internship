class Student:
    def __init__(self, Name, Dept,Rollno,avg=0):
        self.Name = Name
        self.Dept = Dept
        self.Rollno = Rollno
        self.avg=avg
    def Get_Student_Data(self):
        self.Name   = input("Enter the name of the Student       : ")
        self.Dept   = input("Enter the Department of the Student : ")
        self.Rollno =int(input("Enter the Roll No of the Student    : "))
        
    def Print_Student_Data(self):
        print(f"Name of the Student       : {self.Name}")
        print(f"Department of the Student : {self.Dept}")
        print(f"Roll No of the Student    : {self.Rollno}")
        
    def Find_Student_Average(self):
        math = int(input("Enter the marks of SUBJECT (MATH) out of 100   : "))
        sci = int(input("Enter the marks of SUBJECT (SCIENCE) out of 100 : "))
        eng = int(input("Enter the marks of SUBJECT (ENGLISH) out of 100 : "))
        self.avg=math+sci+eng/300
        print(f"Name of the Student    : {self.Name}")        
        print(f"Average of the Student : {self.avg}")

S1=Student("Alex","Computer",1)
S1.Get_Student_Data()
S1.Print_Student_Data()
S1.Find_Student_Average()