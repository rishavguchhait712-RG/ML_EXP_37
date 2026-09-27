import pandas as pd

data = {
    "Student Name": ["Risahb", "Anubhav", "Ankit", "Suvodeep", "Aryan"],
    "Marks": [75, 88, 69, 94, 79]
}

df = pd.DataFrame(data)

def calculate_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"

df["Grade"] = df["Marks"].apply(calculate_grade)

print(df)