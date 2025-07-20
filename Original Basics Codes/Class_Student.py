class Student:
    def __init__(self,name,dept,roll,perc=0):
        self.name=name
        self.dept=dept
        self.roll=roll
        self.perc=perc
        
    def Print_Student_Data(self):
        print()
        print("|------------DETAILS OF STUDENT--------------|")
        print()
        print(f"Name of the Student       : {self.name}")
        print(f"Department of the Student : {self.dept}")
        print(f"Roll No of the Student    : {self.roll}")
        print()
        print("|--------------------------------------------|")
    
    def Find_Student_perc(self):
        print()
        print("|------------PERCENTAGE STUDENT--------------|")
        print()
        math = int(input("Enter the marks of SUBJECT (MATH) out of 100    : "))
        sci = int(input("Enter the marks of SUBJECT (SCIENCE) out of 100 : "))
        eng = int(input("Enter the marks of SUBJECT (ENGLISH) out of 100 : "))
        sum = math+sci+eng
        max_marks = 300
        self.perc=(sum/max_marks)*100
        print(f"Name of the Student       : {self.name}")        
        print(f"Percentage of the Student : {self.perc}")
        print()
        print("|--------------------------------------------|")
        
        
S1=Student("Ben","EJ",10)
S1.Print_Student_Data()
S1.Find_Student_perc()