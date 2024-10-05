"""This file takes in a database, and sorts the information, and gives an improved system of admitting students.
Other files can be put into the algorithm by making it similar to fomat of student_database.csv, and subbing into our filename in the if main statement"""

def check_row_types(row):
    """
    Validates that the row contains exactly 8 elements, all of type float.
    
    Args:
        row (list): List of elements to check.
    
    Returns:
        bool: Returns True if the row is valid, otherwise False.
    """
    if len(row) != 8:
        print("Length incorrect! (should be 8): " + str(row))
        return False
    ind = 0
    while ind < len(row):
        if type(row[ind]) != float:
            print("Type of element incorrect: " + str(row[ind]) + " which is " + str(type(row[ind])))
            return False
        ind += 1
    return True


def convert_row_type(elements):
    """
    Converts a list of string elements to float.
    
    Args:
        elements (list): List of elements to be converted to floats.
    
    Returns:
        list: A new list where all elements have been converted to float.
    """
    sub_converted = []
    for each_string in elements:
        x = float(each_string)
        sub_converted.append(x)
    return sub_converted


def calculate_score(quality_scores):
    """
    Calculates the total score based on the SAT, GPA, interest, and strength of curriculum.
    
    Args:
        quality_scores (list): A list containing [SAT, GPA, interest, strength of curriculum].
    
    Returns:
        float: The calculated total score, rounded to two decimal places.
    """
    sat = quality_scores[0]
    gpa = quality_scores[1]
    interest = quality_scores[2]
    strength_of_curriculum = quality_scores[3]
    total = round(((sat / 160) * 0.3) + ((gpa * 2) * 0.4) + (interest * 0.1) + (strength_of_curriculum * 0.2), 2)
    return total


def is_outlier(quality_scores):
    """
    Determines if a student is an outlier based on the SAT, GPA, and interest.
    
    Args:
        quality_scores (list): A list containing [SAT, GPA, interest].
    
    Returns:
        bool: Returns True if the student is an outlier, otherwise False.
    """
    interest = quality_scores[2]
    gpa = quality_scores[1]
    sat = quality_scores[0]
    normalized_sat = (sat / 160)
    normalized_gpa = (gpa * 2)
    if interest == 0:
        return True
    elif normalized_gpa > (normalized_sat + 2):
        return True
    else:
        return False


def calculate_score_improved(calculated_score, quality_scores):
    """
    Determines if the student should be considered improved based on their score and outlier status.
    
    Args:
        calculated_score (float): The student's calculated total score.
        quality_scores (list): A list containing [SAT, GPA, interest, strength of curriculum].
    
    Returns:
        bool: True if the score is 6 or above, or the student is an outlier. False otherwise.
    """
    if calculated_score >= 6:
        return True
    elif is_outlier(quality_scores):
        return True
    else:
        return False


def grade_outlier(semester_grades):
    """
    Checks if there is a significant difference (greater than 20) between the first two sorted semester grades.
    
    Args:
        semester_grades (list): A list of four semester grades.
    
    Returns:
        bool: True if the difference between the first two grades is greater than 20, otherwise False.
    """
    sorted_grades = sorted(semester_grades)
    difference = sorted_grades[1] - sorted_grades[0]
    if difference > 20:
        return True
    else:
        return False


def grade_improvement(semester_grades):
    """
    Checks if the student's grades improve or remain the same across four semesters.
    
    Args:
        semester_grades (list): A list of four semester grades.
    
    Returns:
        bool: True if the grades are in non-decreasing order, otherwise False.
    """
    if semester_grades[3] >= semester_grades[2] >= semester_grades[1] >= semester_grades[0]:
        return True
    else:
        return False


def main(filename):
    """
    Processes student data from a CSV file and writes output to multiple files based on score, outlier status,
    and grade improvement.
    
    The input file 'superheroes_tiny.csv' contains student data. Outputs are written to various files:
    - student_scores.csv: All students' names and calculated scores.
    - chosen_students.txt: Students with a score of 6 or more.
    - outliers.txt: Students identified as outliers.
    - chosen_improved.txt: Students with score >= 6 or considered as improved.
    - improved_chosen.csv: Students considered improved with detailed scores.
    - extra_improved_chosen.txt: Students meeting extra improvement criteria.
    """
    input_file = open(filename, "r")    
    print("Processing " + filename + "...")
    headers = input_file.readline()  # Take a header, so we don't iterate through the first line. Variable will be unused
    the_rest = input_file.readlines()  # Read the remaining lines (students' data)
    input_file.close()
    
    # Open output files for writing processed data
    student_scores_files = open('student_scores.csv', "w")
    chosen_students_file = open('chosen_students.txt', "w")
    outliers_file = open('outliers.txt', "w")
    chosen_improved_file = open('chosen_improved.txt', "w")
    improved_chosen_csv = open('improved_chosen.csv', "w")
    extra_improved_chosen_file = open('extra_improved_chosen.txt', "w")
    
    # Process each line of relevant student data
    for line in the_rest:
        elements = line.split(',')  # Split the line by commas to separate data fields
        student_names = elements[0]  # Extract the student's name
        elements = elements[1:]  # Get the remaining elements (scores and grades)
        converted = convert_row_type(elements)  # Convert the scores and grades from strings to floats
        check_row_types(converted)  # Validate the converted data
        
        quality_scores = converted[0:4]  # Extract quality scores (SAT, GPA, etc.)
        semester_grades = converted[4:8]  # Extract semester grades
        
        calculated_score = calculate_score(quality_scores)  # Calculate the student's score
        
        student_scores_files.write(f"{student_names},{calculated_score:.2f}\n")  # Write score to file
        
        if calculated_score >= 6:
            chosen_students_file.write(f"{student_names}\n")  # Write student to chosen list if score >= 6
        
        if is_outlier(quality_scores):
            outliers_file.write(f"{student_names}\n")  # Write to outliers if the student is an outlier
        
        if calculated_score >= 6:
            chosen_improved_file.write(f"{student_names}\n")  # Write to improved list if score >= 6
        elif is_outlier(quality_scores) and calculated_score >= 5:
            chosen_improved_file.write(f"{student_names}\n")  # Write to improved list if outlier and score >= 5
        
        # Write improved students to CSV with detailed scores
        if calculate_score_improved(calculated_score, quality_scores):
            sat = quality_scores[0]
            gpa = quality_scores[1]
            interest = quality_scores[2]
            strength_of_curriculum = quality_scores[3]
            improved_chosen_csv.write(f"{student_names},{sat},{gpa},{interest},{strength_of_curriculum}\n")
        
        # Additional criteria for extra improvement
        if calculated_score >= 6:
            extra_improved_chosen_file.write(f"{student_names}\n")
        elif calculated_score >= 5 and is_outlier(quality_scores):
            extra_improved_chosen_file.write(f"{student_names}\n")
        elif calculated_score >= 5 and grade_outlier(semester_grades):
            extra_improved_chosen_file.write(f"{student_names}\n")
        elif calculated_score >= 5 and grade_improvement(semester_grades):
            extra_improved_chosen_file.write(f"{student_names}\n")
    
    # Close all output files
    student_scores_files.close()
    chosen_students_file.close()
    outliers_file.close()
    chosen_improved_file.close()
    improved_chosen_csv.close()
    extra_improved_chosen_file.close()
    
    print("Process Finished")


if __name__ == "__main__":
    filename = "student_database.csv"
    main(filename)