from flask import Flask, render_template, request, redirect, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Use a temporary directory for file uploads
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/analyze')
def index():
    return render_template('index.html')

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        try:
            # Read the file directly from the request into pandas
            df = pd.read_excel(file)
            
            # Process the Excel data
            graph_url, best_strand, student_data = process_excel(df)
            
            if graph_url is None:
                return 'Invalid Excel file format. Please ensure it contains the required columns.'

            return render_template('index.html', graph_url=graph_url, best_strand=best_strand, student_data=student_data)
        except Exception as e:
            return f'Error processing file: {str(e)}'

    return 'File type not allowed or no file selected.'

def process_excel(df):
    try:
        # Create a clean dataset with student numbers and grades
        student_data = []
        
        # Skip the first row (header)
        for idx, row in df.iloc[1:].iterrows():
            # Skip if student number is empty or NaN
            if pd.isna(row.iloc[0]) or str(row.iloc[0]).strip() == '':
                continue
                
            student_info = {
                'Student Number': row.iloc[0],  # First column contains student number
                'Grade 7': {
                    'English': float(row.iloc[1]),  # Grade 7 English
                    'Filipino': float(row.iloc[2]),  # Grade 7 Filipino
                    'Math': float(row.iloc[3]),  # Grade 7 Math
                    'Science': float(row.iloc[4]),  # Grade 7 Science
                    'Mapeh': float(row.iloc[5]),  # Grade 7 Mapeh
                    'TLE': float(row.iloc[6])  # Grade 7 TLE
                },
                'Grade 8': {
                    'English': float(row.iloc[7]),  # Grade 8 English
                    'Filipino': float(row.iloc[8]),  # Grade 8 Filipino
                    'Math': float(row.iloc[9]),  # Grade 8 Math
                    'Science': float(row.iloc[10]),  # Grade 8 Science
                    'Mapeh': float(row.iloc[11]),  # Grade 8 Mapeh
                    'TLE': float(row.iloc[12])  # Grade 8 TLE
                },
                'Grade 9': {
                    'English': float(row.iloc[13]),  # Grade 9 English
                    'Filipino': float(row.iloc[14]),  # Grade 9 Filipino
                    'Math': float(row.iloc[15]),  # Grade 9 Math
                    'Science': float(row.iloc[16]),  # Grade 9 Science
                    'Mapeh': float(row.iloc[17]),  # Grade 9 Mapeh
                    'TLE': float(row.iloc[18])  # Grade 9 TLE
                },
                'Grade 10': {
                    'English': float(row.iloc[19]),  # Grade 10 English
                    'Filipino': float(row.iloc[20]),  # Grade 10 Filipino
                    'Math': float(row.iloc[21]),  # Grade 10 Math
                    'Science': float(row.iloc[22]),  # Grade 10 Science
                    'Mapeh': float(row.iloc[23]),  # Grade 10 Mapeh
                    'TLE': float(row.iloc[24])  # Grade 10 TLE
                }
            }
            
            # Calculate average grades per subject across all grade levels
            student_info['Average Grades'] = {
                'English': sum(student_info[grade]['English'] for grade in ['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']) / 4,
                'Filipino': sum(student_info[grade]['Filipino'] for grade in ['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']) / 4,
                'Science': sum(student_info[grade]['Science'] for grade in ['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']) / 4,
                'Math': sum(student_info[grade]['Math'] for grade in ['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']) / 4,
                'Mapeh': sum(student_info[grade]['Mapeh'] for grade in ['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']) / 4,
                'TLE': sum(student_info[grade]['TLE'] for grade in ['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']) / 4
            }
            
            student_data.append(student_info)

        # Define grade thresholds for each strand
        strand_subjects = {
            'STEM': ['Math', 'Science'],
            'HUMSS': ['English', 'Filipino'],
            'ABM': ['Math', 'English'],
            'TVL': ['TLE', 'Mapeh']
        }

        # Calculate strand scores for each student based on average grades
        for student in student_data:
            student_scores = {}
            for strand, subjects in strand_subjects.items():
                avg = sum(student['Average Grades'][subject] for subject in subjects) / len(subjects)
                student_scores[strand] = avg
            student['Recommended Strand'] = max(student_scores, key=student_scores.get)
            student['Strand Scores'] = student_scores

        # Calculate overall class performance for each strand
        strand_scores = {strand: 0 for strand in strand_subjects.keys()}
        for student in student_data:
            for strand in strand_scores:
                strand_scores[strand] += student['Strand Scores'][strand]
        
        # Calculate averages
        num_students = len(student_data)
        for strand in strand_scores:
            strand_scores[strand] /= num_students

        # Find the best overall strand
        best_strand = max(strand_scores, key=strand_scores.get)

        # Create a bar graph for the strand strengths
        plt.figure(figsize=(12, 6))
        bars = plt.bar(strand_scores.keys(), strand_scores.values(), color=['#2196F3', '#4CAF50', '#FFC107', '#F44336'])
        plt.xlabel('Strands')
        plt.ylabel('Average Grade')
        plt.title('Average Performance by Strand (Grades 7-10)')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        # Save the graph to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        
        # Convert the image to base64 for embedding in HTML
        graph_url = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        # Prepare simplified student data for display
        display_data = []
        for student in student_data:
            display_info = {
                'Student Number': student['Student Number'],
                'Average English': f"{student['Average Grades']['English']:.1f}",
                'Average Filipino': f"{student['Average Grades']['Filipino']:.1f}",
                'Average Science': f"{student['Average Grades']['Science']:.1f}",
                'Average Math': f"{student['Average Grades']['Math']:.1f}",
                'Average Mapeh': f"{student['Average Grades']['Mapeh']:.1f}",
                'Average TLE': f"{student['Average Grades']['TLE']:.1f}",
                'Recommended Strand': student['Recommended Strand'],
                'STEM Score': f"{student['Strand Scores']['STEM']:.1f}",
                'HUMSS Score': f"{student['Strand Scores']['HUMSS']:.1f}",
                'ABM Score': f"{student['Strand Scores']['ABM']:.1f}",
                'TVL Score': f"{student['Strand Scores']['TVL']:.1f}"
            }
            display_data.append(display_info)

        return f"data:image/png;base64,{graph_url}", best_strand, display_data
    except Exception as e:
        print(f"Error in process_excel: {str(e)}")
        return None, None, None
    
@app.route('/dash')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/predict_grades', methods=['POST'])
def predict_grades():
    try:
        data = request.get_json()
        student_grades = pd.DataFrame(data['grades'])
        
        # Prepare data for linear regression
        X = []  # Features (previous grades)
        y = []  # Target (next grade)
        
        # For each subject, create a prediction model
        subjects = ['English', 'Filipino', 'Math', 'Science', 'Mapeh', 'TLE']
        predictions = {}
        
        for subject in subjects:
            grades = student_grades[subject].values
            
            # Create features and target
            for i in range(len(grades)-1):
                X.append([grades[i]])
                y.append(grades[i+1])
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Create and train the model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict next grade based on last grade
            last_grade = grades[-1]
            next_grade_prediction = model.predict([[last_grade]])[0]
            
            predictions[subject] = {
                'current_grade': float(last_grade),
                'predicted_next_grade': float(next_grade_prediction),
                'trend': 'up' if next_grade_prediction > last_grade else 'down'
            }
            
            # Clear X and y for next subject
            X = []
            y = []
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/student_stats')
def get_student_stats():
    try:
        # Get all student data from your database or files
        # For now, we'll return dummy data
        stats = {
            'total_students': 150,
            'average_grades': {
                'English': 85.5,
                'Math': 82.3,
                'Science': 84.1,
                'Filipino': 86.7,
                'Mapeh': 88.2,
                'TLE': 87.9
            },
            'strand_distribution': {
                'STEM': 35,
                'HUMSS': 28,
                'ABM': 45,
                'TVL': 42
            }
        }
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
