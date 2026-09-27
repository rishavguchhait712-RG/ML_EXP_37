import pandas as pd

data = {
    "Student Name": ["Rishab", "Anubhav", "Ankit", "Suvodeep", "Aryan"],
    "Roll Number": [37, 11, 7, 58, 13],
    "Marks": [75, 88, 69, 94, 79],
    "Attendance": [85, 91, 78, 96, 83]
}

df = pd.DataFrame(data)

print("--- Complete Student Data ---")
print(df)

print("\n--- Students Scoring Above 80 ---")
print(df[df["Marks"] > 80])
