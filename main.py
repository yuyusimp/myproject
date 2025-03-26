from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid threading issues
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import io
import base64
import tempfile
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from psycopg2 import pool
from functools import wraps
import datetime
import json
import random
from io import BytesIO
import xlsxwriter
from flask import send_file

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = datetime.timedelta(minutes=30)  # Session expires after 30 minutes

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'strand_classifier.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
MLR_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'multinomial_logistic_regression.pkl')

# Define DepEd criteria for strands based on official DepEd Philippines guidelines
STRAND_CRITERIA = {
    'STEM': {
        'required_grades': {
            'Math': 85,
            'Science': 85
        },
        'assessment_score': 86
    },
    'ABM': {},  # No specific requirements
    'HUMSS': {},  # No specific requirements
    'TVL': {},  # No specific requirements
    'GAS': {}  # No specific requirements
}

# Load the trained models
def load_model():
    """Load the strand classification model"""
    try:
        # Try loading the joblib model first
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            return model
        # If not found, try the MLR model
        elif os.path.exists(MLR_MODEL_PATH):
            print(f"Loading model from {MLR_MODEL_PATH}")
            model = joblib.load(MLR_MODEL_PATH)
            return model
        else:
            # Fallback to a basic model
            print("No saved models found. Using fallback model.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Create sample data to fit the model
            X = np.random.rand(100, 20)  # 20 features for 10 subjects with raw and average grades
            y = np.random.choice(['STEM', 'HUMSS', 'ABM', 'TVL', 'GAS'], size=100)
            
            # Fit the model
            model.fit(X, y)
            return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def load_scaler():
    """Load the feature scaler or create a new one if not found"""
    try:
        if os.path.exists(SCALER_PATH):
            print(f"Loading scaler from {SCALER_PATH}")
            scaler = joblib.load(SCALER_PATH)
            return scaler
        else:
            print("No saved scaler found. Creating a new scaler.")
            from sklearn.preprocessing import StandardScaler
            
            # Create a new scaler and fit it with reasonable education data
            # Generate sample data in the range of typical grades (0-100)
            sample_data = np.random.uniform(60, 100, size=(100, 20))
            
            # Create and fit the scaler
            scaler = StandardScaler()
            scaler.fit(sample_data)
            
            # Save the scaler for future use
            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            joblib.dump(scaler, SCALER_PATH)
            
            print("New scaler created and saved.")
            return scaler
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        raise

def prepare_features(grades):
    """Preprocess student grades for model prediction"""
    # Expected features in the correct order - each subject has two features (their raw grade and average)
    core_subjects = ['English', 'Filipino', 'Math', 'Science', 'ESP', 'ICT', 'TLE', 'AP', 'Mapeh', 'Tec Drawing']
    
    # Create feature vector
    features = []
    
    # First add the raw grades for each subject
    for subject in core_subjects:
        features.append(float(grades.get(subject, 0)))
    
    # Then add the averages for each subject (this creates the full 20 features the model expects)
    for subject in core_subjects:
        features.append(float(grades.get(f'Average {subject}', grades.get(subject, 0))))
    
    # Ensure we have exactly 20 features
    if len(features) != 20:
        print(f"Warning: Expected 20 features, but got {len(features)}")
        # Pad with zeros if less than 20
        features.extend([0] * (20 - len(features)))
        # Truncate if more than 20
        features = features[:20]
    
    # Try to use the scaler if available
    try:
        scaler = load_scaler()
        if hasattr(scaler, 'transform'):
            features = scaler.transform([features])[0]
        else:
            # Simple normalization
            features = np.array(features) / 100.0
    except Exception as e:
        print(f"Error during scaling: {str(e)}")
        # Simple normalization as fallback
        features = np.array(features) / 100.0
    
    return features

def check_strand_eligibility(grades, assessment_scores):
    """Check which strands a student is eligible for based on official DepEd criteria"""
    eligible_strands = []
    
    # Check STEM eligibility - requires 85+ in both Math and Science, 86+ percentile in assessment
    stem_criteria = STRAND_CRITERIA['STEM']
    stem_eligible = True
    
    for subject, min_grade in stem_criteria.get('required_grades', {}).items():
        avg_subject = f'Average {subject}'
        subject_grade = grades.get(avg_subject, 0)
        if subject_grade == 0:  # If average not available, use the direct grade
            subject_grade = grades.get(subject, 0)
            
        if subject_grade < min_grade:
            print(f"Student not eligible for STEM due to {subject} grade: {subject_grade} < {min_grade}")
            stem_eligible = False
            break
    
    stem_assessment = assessment_scores.get('STEM', 0)
    if stem_eligible:
        if stem_assessment >= stem_criteria.get('assessment_score', 86):
            eligible_strands.append('STEM')
            print(f"Student is eligible for STEM with grades and assessment: {stem_assessment}")
        else:
            print(f"Student has grades for STEM but assessment score too low: {stem_assessment} < 86")
            eligible_strands.append('STEM (Subject to assessment score)')
    
    # Other strands (ABM, HUMSS, TVL, GAS) have no specific grade or assessment requirements
    for strand in ['ABM', 'HUMSS', 'TVL', 'GAS']:
        eligible_strands.append(strand)
    
    return eligible_strands

def process_excel(df, filename="Excel Upload"):
    """Process the uploaded Excel file and predict strands"""
    try:
        # Load the trained model
        model = load_model()
        
        # Print DataFrame info for debugging
        print("DataFrame Info:")
        print(df.info())
        print("\nDataFrame Columns:", df.columns.tolist())
        
        # Check if we have a multi-level structure (headers in the data)
        is_complex_format = False
        if 'Unnamed: 0' in df.columns and 'Grade' in str(df.columns):
            is_complex_format = True
            print("Detected complex format with headers in data rows")
            
            # Get header row (first row - index 0)
            header_row = df.iloc[0]
            print("Header row:", header_row.tolist())
            
            # Find subjects in header row
            subjects = ['English', 'Filipino', 'Math', 'Science', 'ESP', 'ICT', 'TLE', 'AP', 'Mapeh', 'Tec Drawing']
            
            # Field name mappings for different naming conventions
            field_mappings = {
                'ICT': ['ICF', 'ICT', 'Information Computer Fundamental'],  # ICF stands for Information Computer Fundamental
                'TLE': ['TVL', 'TVE', 'TLE'],  # TVL instead of TVE 
                'Tec Drawing': ['Tec. Drawing', 'Tec Drawing', 'Technical Drawing']
            }
            
            # Clean up column mappings
            col_mappings = {}
            for col_idx, value in enumerate(header_row):
                if pd.notna(value) and isinstance(value, str):
                    value = value.strip()
                    for subject in subjects:
                        # Normalize name comparison
                        subject_normalized = subject.lower().replace('.', '')
                        value_normalized = value.lower().replace('.', '')
                        
                        # Check for direct match or matches in field mappings
                        matches = False
                        if subject_normalized in value_normalized:
                            matches = True
                        elif subject in field_mappings:
                            # Check alternate names
                            for alt_name in field_mappings[subject]:
                                if alt_name.lower().replace('.', '') in value_normalized:
                                    matches = True
                                    break
                        
                        if matches:
                            # Check if this is Grade 10 or the last occurrence
                            grade_col = None
                            for i in range(col_idx, 0, -1):
                                if 'Grade' in str(df.columns[i]) and '10' in str(df.columns[i]):
                                    grade_col = i
                                    break
                            
                            if grade_col is not None:
                                print(f"Found {subject} for Grade 10 at column {col_idx}")
                                col_mappings[subject] = col_idx
                            elif subject not in col_mappings:
                                # If we haven't found this subject yet, use this column
                                print(f"Found {subject} at column {col_idx}")
                                col_mappings[subject] = col_idx
            
            print("Column mappings:", col_mappings)
            
            # Extract student data
            student_data = []
            for i in range(1, len(df)):  # Start from row 1 (second row) to skip header
                try:
                    student_row = df.iloc[i]
                    student_id = student_row[0]  # Student ID in first column
                    
                    if pd.isna(student_id) or str(student_id).strip() == '':
                        continue
                    
                    student_dict = {'Student No': str(student_id)}
                    
                    # Process each subject
                    for subject, col_idx in col_mappings.items():
                        if col_idx < len(student_row):
                            grade_value = student_row[col_idx]
                            # Try to convert to float
                            try:
                                if isinstance(grade_value, str):
                                    grade_value = grade_value.replace('%', '').strip()
                                
                                if pd.notna(grade_value) and grade_value != '':
                                    student_dict[subject] = float(grade_value)
                                else:
                                    student_dict[subject] = 0.0
                            except (ValueError, TypeError):
                                print(f"Error converting {grade_value} to float for {subject}")
                                student_dict[subject] = 0.0
                        else:
                            student_dict[subject] = 0.0
                    
                    # Calculate average grades for subjects
                    for subject in set(col_mappings.keys()):
                        avg_key = f'Average {subject}'
                        student_dict[avg_key] = student_dict.get(subject, 0)
                    
                    # Generate mock assessment scores for demonstration
                    # In a real application, these would come from actual assessment data
                    assessment_scores = {
                        'STEM': random.randint(70, 99),
                        'HUMSS': random.randint(70, 99),
                        'ABM': random.randint(70, 99),
                        'TVL': random.randint(70, 99),
                        'GAS': random.randint(70, 99)
                    }
                    
                    # Add assessment scores to student dictionary for reference
                    student_dict.update(assessment_scores)
                    
                    # Prepare features for prediction
                    features = prepare_features(student_dict)
                    
                    # Predict strand using the model
                    try:
                        strand_prediction = model.predict([features])[0]
                        probabilities = model.predict_proba([features])[0]
                        
                        # Map numeric prediction to strand name
                        strand_mapping = {0: 'ABM', 1: 'STEM', 2: 'TVL', 3: 'HUMSS', 4: 'GAS'}
                        predicted_strand = strand_mapping.get(strand_prediction, 'Unknown')
                        
                        # Store strand probabilities as percentages
                        for i, prob in enumerate(probabilities):
                            strand = strand_mapping.get(i, 'Unknown')
                            student_dict[f'{strand} Score'] = round(prob * 100, 2)
                        
                        # Get eligible strands based on DepEd criteria
                        eligible_strands = check_strand_eligibility(student_dict, assessment_scores)
                        
                        # If predicted strand is not in eligible strands, adjust prediction
                        # For example, if model predicts STEM but student doesn't meet criteria
                        if predicted_strand not in eligible_strands and not any(predicted_strand in s for s in eligible_strands):
                            # Find the most probable eligible strand
                            max_prob = -1
                            for i, prob in enumerate(probabilities):
                                strand = strand_mapping.get(i, 'Unknown')
                                if strand in eligible_strands and prob > max_prob:
                                    max_prob = prob
                                    predicted_strand = strand
                            
                            # If still no match, default to most appropriate eligible strand
                            if predicted_strand not in eligible_strands and not any(predicted_strand in s for s in eligible_strands):
                                # Default to ABM as it has no specific requirements
                                predicted_strand = 'ABM'
                                print(f"Adjusted strand from {strand_prediction} to {predicted_strand} based on eligibility")
                        
                        # Add strand prediction to student dictionary
                        student_dict['Predicted Strand'] = predicted_strand
                        student_dict['Eligible Strands'] = eligible_strands
                        
                    except Exception as e:
                        print(f"Error during prediction: {str(e)}")
                        student_dict['Predicted Strand'] = 'Error'
                        student_dict['Eligible Strands'] = []
                    
                    student_data.append(student_dict)
                except Exception as e:
                    print(f"Error processing student row {i}: {e}")
            
            print(f"Processed {len(student_data)} students from complex format")
            
        # If not complex or no students processed, try standard format
        if not is_complex_format or len(student_data) == 0:
            # Determine if this is the sample dataset format
            expected_columns = ['Student No', 'English', 'Filipino', 'Math', 'Science', 'ESP', 'ICT', 'TLE', 'AP', 'Mapeh', 'Tec Drawing']
            
            # Check if the DataFrame has the expected columns
            if all(col in df.columns for col in expected_columns):
                print("Using standard format with subject columns")
                data_df = df
                is_sample_format = True
            elif len(df.columns) > 10 and 'Unnamed: 0' in df.columns:
                # This might be the complex format with grade levels
                print("Complex format detected, processing...")
                
                # Extract student numbers (usually in first column)
                student_nos = df.iloc[1:, 0].tolist()
                
                # Create a new DataFrame with proper structure
                data = []
                subjects = ['English', 'Filipino', 'Math', 'Science', 'ESP', 'ICT', 'TLE', 'AP', 'Mapeh', 'Tec Drawing']
                
                for i, student_no in enumerate(student_nos):
                    if pd.isna(student_no) or student_no == '':
                        continue
                    
                    student_row = {'Student No': student_no}
                    
                    # Process each subject by taking the Grade 10 value if available
                    for subject in subjects:
                        # Find columns that match this subject
                        subject_cols = [col for col in df.columns if subject in str(col)]
                        if subject_cols:
                            # Try to find Grade 10 values first
                            grade10_cols = [col for col in subject_cols if '10' in str(col)]
                            if grade10_cols:
                                # Use the first Grade 10 column found
                                student_row[subject] = df.iloc[i+1, df.columns.get_loc(grade10_cols[0])]
                            else:
                                # If no Grade 10, use the first available column for this subject
                                student_row[subject] = df.iloc[i+1, df.columns.get_loc(subject_cols[0])]
                        else:
                            # If subject not found, set to None
                            student_row[subject] = None
                    
                    data.append(student_row)
                
                # Convert to DataFrame
                data_df = pd.DataFrame(data)
                print("Created new DataFrame with structure:", data_df.columns.tolist())
            else:
                # Try to intelligently map columns
                print("Attempting to map columns intelligently...")
                
                # Try to find student number column
                student_no_col = None
                for col in df.columns:
                    if 'student' in str(col).lower() and ('no' in str(col).lower() or 'number' in str(col).lower()):
                        student_no_col = col
                        break
                
                if not student_no_col:
                    # If we can't find a clear student number column, use the first column
                    student_no_col = df.columns[0]
                
                # Extract mapping of subjects
                column_mapping = {}
                for expected_col in expected_columns[1:]:  # Skip Student No
                    for actual_col in df.columns:
                        if expected_col.lower() in str(actual_col).lower():
                            column_mapping[expected_col] = actual_col
                            break
                
                # Create a new DataFrame with mapped columns
                data = []
                for _, row in df.iterrows():
                    student_row = {'Student No': row[student_no_col]}
                    
                    for expected_col, actual_col in column_mapping.items():
                        student_row[expected_col] = row[actual_col]
                    
                    data.append(student_row)
                
                data_df = pd.DataFrame(data)
                print("Created mapped DataFrame with structure:", data_df.columns.tolist())
                
                # If we need to process more student data
                if is_complex_format and len(student_data) > 0:
                    # Skip standard processing
                    pass
                else:
                    # Clean the data - convert grades to float and handle missing values
                    student_data = []
                    for _, row in data_df.iterrows():
                        student_dict = {'Student No': row['Student No']}
                        
                        # Process each subject
                        for subject in expected_columns[1:]:
                            if subject in row:
                                try:
                                    # Try to convert to float, handle various formats
                                    grade_str = str(row[subject]).replace('%', '').strip()
                                    student_dict[subject] = float(grade_str) if grade_str and not pd.isna(grade_str) else 0
                                except (ValueError, TypeError):
                                    student_dict[subject] = 0
                            else:
                                student_dict[subject] = 0
                        
                        # Calculate averages for key subjects
                        for subject in expected_columns[1:]:
                            student_dict[f'Average {subject}'] = student_dict[subject]
                        
                        # Generate mock assessment scores for demonstration
                        # In a real application, these would come from actual assessment data
                        assessment_scores = {
                            'STEM': random.randint(70, 99),
                            'HUMSS': random.randint(70, 99),
                            'ABM': random.randint(70, 99),
                            'TVL': random.randint(70, 99),
                            'GAS': random.randint(70, 99)
                        }
                        
                        # Add assessment scores to student dictionary for reference
                        student_dict.update(assessment_scores)
                        
                        # Prepare features for prediction
                        features = prepare_features(student_dict)
                        
                        # Predict strand using the model
                        try:
                            strand_prediction = model.predict([features])[0]
                            probabilities = model.predict_proba([features])[0]
                            
                            # Map numeric prediction to strand name
                            strand_mapping = {0: 'ABM', 1: 'STEM', 2: 'TVL', 3: 'HUMSS', 4: 'GAS'}
                            predicted_strand = strand_mapping.get(strand_prediction, 'Unknown')
                            
                            # Store strand probabilities as percentages
                            for i, prob in enumerate(probabilities):
                                strand = strand_mapping.get(i, 'Unknown')
                                student_dict[f'{strand} Score'] = round(prob * 100, 2)
                            
                            # Get eligible strands based on DepEd criteria
                            eligible_strands = check_strand_eligibility(student_dict, assessment_scores)
                            
                            # If predicted strand is not in eligible strands, adjust prediction
                            # For example, if model predicts STEM but student doesn't meet criteria
                            if predicted_strand not in eligible_strands and not any(predicted_strand in s for s in eligible_strands):
                                # Find the most probable eligible strand
                                max_prob = -1
                                for i, prob in enumerate(probabilities):
                                    strand = strand_mapping.get(i, 'Unknown')
                                    if strand in eligible_strands and prob > max_prob:
                                        max_prob = prob
                                        predicted_strand = strand
                                
                                # If still no match, default to most appropriate eligible strand
                                if predicted_strand not in eligible_strands and not any(predicted_strand in s for s in eligible_strands):
                                    # Default to ABM as it has no specific requirements
                                    predicted_strand = 'ABM'
                                    print(f"Adjusted strand from {strand_prediction} to {predicted_strand} based on eligibility")
                            
                            # Add strand prediction to student dictionary
                            student_dict['Predicted Strand'] = predicted_strand
                            student_dict['Eligible Strands'] = eligible_strands
                            
                        except Exception as e:
                            print(f"Error during prediction: {str(e)}")
                            student_dict['Predicted Strand'] = 'Error'
                            student_dict['Eligible Strands'] = []
                        
                        student_data.append(student_dict)
        
        # Debug info
        print(f"Processed {len(student_data)} students")
        if student_data:
            print("Sample student data:", student_data[0])
            
        # Plot the data
        plt.figure(figsize=(15, 6))
        
        # 1. Count of Recommended Strands (Left)
        plt.subplot(1, 2, 1)
        
        # Count strands
        strand_counts = {}
        for student in student_data:
            strand = student['Predicted Strand']
            strand_counts[strand] = strand_counts.get(strand, 0) + 1
        
        # Sort strands for consistency
        sorted_strands = sorted(strand_counts.keys())
        counts = [strand_counts.get(strand, 0) for strand in sorted_strands]
        
        # Define colors for each strand
        strand_colors = {
            'STEM': '#3498db',   # Blue
            'HUMSS': '#e74c3c',  # Red
            'ABM': '#2ecc71',    # Green
            'TVL': '#f39c12',    # Orange
            'GAS': '#1abc9c'  # Teal
        }
        
        # Get colors for the bars
        bar_colors = [strand_colors.get(strand, '#95a5a6') for strand in sorted_strands]
        
        # Create the bar chart
        bars = plt.bar(sorted_strands, counts, color=bar_colors)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.xlabel('Strands', fontsize=10)
        plt.ylabel('Number of Students', fontsize=10)
        plt.title('Distribution of Recommended Strands', fontsize=12, fontweight='bold')
        plt.ylim(0, max(counts) + 5)  # Add some space for the labels
        plt.xticks(rotation=0)
        
        # 2. Average Grades by Strand (Right)
        plt.subplot(1, 2, 2)
        
        # Calculate average grades per strand
        strand_subject_averages = {}
        
        # Initialize the data structure
        for strand in sorted_strands:
            strand_subject_averages[strand] = {}
            for subject in ['English', 'Filipino', 'Math', 'Science', 'ESP']:
                strand_subject_averages[strand][subject] = []
        
        # Collect grades by strand
        for student in student_data:
            strand = student['Predicted Strand']
            if strand in strand_subject_averages:
                for subject in ['English', 'Filipino', 'Math', 'Science', 'ESP']:
                    grade = student.get(subject, 0)
                    strand_subject_averages[strand][subject].append(grade)
        
        # Calculate averages
        strand_avg_data = {}
        for strand in strand_subject_averages:
            strand_avg_data[strand] = {}
            for subject in strand_subject_averages[strand]:
                grades = strand_subject_averages[strand][subject]
                if grades:
                    strand_avg_data[strand][subject] = sum(grades) / len(grades)
                else:
                    strand_avg_data[strand][subject] = 0
        
        # Set up grouped bar chart
        bar_width = 0.15
        index = np.arange(len(sorted_strands))
        
        # Plot bars for each subject
        subjects_to_show = ['English', 'Filipino', 'Math', 'Science', 'ESP']
        subject_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, subject in enumerate(subjects_to_show):
            values = [strand_avg_data.get(strand, {}).get(subject, 0) for strand in sorted_strands]
            plt.bar(index + i * bar_width, values, bar_width, label=subject, color=subject_colors[i])
        
        plt.xlabel('Strands', fontsize=10)
        plt.ylabel('Average Grade', fontsize=10)
        plt.title('Average Grades by Strand', fontsize=12, fontweight='bold')
        plt.xticks(index + bar_width * 2, sorted_strands)
        plt.legend(loc='upper right', ncol=5, fontsize='small')
        plt.ylim(0, 100)
        
        plt.tight_layout(pad=3)
        
        # Save the plot to a BytesIO object instead of a file
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Create a directory for plots if it doesn't exist
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        plot_filename = f"{secure_filename(filename.split('.')[0])}_{timestamp}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        
        # Save the plot to file
        with open(plot_path, 'wb') as f:
            f.write(buf.getvalue())
            
        # Return the file path relative to the static folder
        rel_path = os.path.join('plots', plot_filename)
        
        print(f"Plot saved to: {plot_path}")
        print(f"Returning path: {rel_path} and {len(student_data)} student records")
        
        return student_data, rel_path
        
    except Exception as e:
        print(f"Error in process_excel: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Database connection
conn = psycopg2.connect(
    dbname="postgres", user="postgres",
    password="2002", host="localhost"
)
cur = conn.cursor()

db_pool = pool.SimpleConnectionPool(
    1, 10,  # min and max connections
    dbname="postgres", user="postgres",
    password="2002", host="localhost"
)

# Function to get a database connection
def get_db_conn():
    return db_pool.getconn()

# Function to release a connection back to the pool
def release_db_conn(conn):
    db_pool.putconn(conn)

# Function to set up the database tables
def setup_database():
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Create users table if it doesn't exist
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(200) NOT NULL,
                role VARCHAR(50) DEFAULT 'user'
            )
        ''')
        
        # Create transaction_logs table if it doesn't exist
        cur.execute('''
            CREATE TABLE IF NOT EXISTS transaction_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(user_id),
                filename VARCHAR(255),
                num_students INTEGER,
                best_strand VARCHAR(50),
                strand_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add strand_data column if it doesn't exist
        try:
            cur.execute('''
                ALTER TABLE transaction_logs 
                ADD COLUMN IF NOT EXISTS strand_data TEXT
            ''')
        except:
            pass  # Ignore error if column already exists
        
        conn.commit()
        cur.close()
        release_db_conn(conn)
        print("Database setup completed.")
        
    except Exception as e:
        print(f"Error setting up database: {e}")

# Use a temporary directory for file uploads
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def destroy_session():
    session.clear()
    flash("You have been logged out.", "success")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Add cache control headers to prevent caching
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/analyze')
@login_required
def index():
    return render_template('index.html')

@app.route('/')
def login():
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Connect to database
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Check if user exists and password is correct
        cur.execute('SELECT user_id, password_hash FROM users WHERE username = %s', (username,))
        user = cur.fetchone()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['login_time'] = datetime.datetime.now().timestamp()
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
            return redirect(url_for('login'))
        
        cur.close()
        release_db_conn(conn)
    
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('adduser.html')

@app.route('/add', methods=['POST'])
def adduser():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = get_db_conn()
        cur = conn.cursor()

        try:
            cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                        (username, email, hashed_password))
            conn.commit()
            flash('User added successfully!', 'success')
            return redirect(url_for('view_users'))
        except Exception as e:
            conn.rollback()
            flash(f"Error adding user: {str(e)}", 'danger')
        finally:
            cur.close()
            release_db_conn(conn)

        return redirect(url_for('view_users'))

@app.route('/users')
@login_required
def view_users():
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        # Get search parameters
        username = request.args.get('username', '')
        email = request.args.get('email', '')

        # Build the query
        query = "SELECT user_id, username, email, created_at FROM users WHERE 1=1"
        params = []

        if username:
            query += " AND username ILIKE %s"
            params.append(f"%{username}%")
        if email:
            query += " AND email ILIKE %s"
            params.append(f"%{email}%")

        query += " ORDER BY created_at DESC"
        
        cur.execute(query, params)
        users = cur.fetchall()
        return render_template('viewusers.html', users=users)
    except Exception as e:
        flash(f"Error retrieving users: {str(e)}", 'danger')
        return render_template('viewusers.html', users=[])
    finally:
        cur.close()
        release_db_conn(conn)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    conn.commit()
    return redirect(url_for('view_users'))

@app.route('/update_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def update_user(user_id):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    release_db_conn(conn)
    return render_template('update_user.html', user = user)

@app.route('/update_user1', methods=['POST'])
@login_required
def update_user1():
    if request.method == "POST":
        user_id = request.form['user_id']
        username = request.form['username']
        email = request.form['email']
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("UPDATE users SET username = %s, email = %s WHERE user_id = %s", (username, email, user_id))
        conn.commit()
        return redirect(url_for('view_users'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            try:
                # Save to temp file
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(temp_path)
                
                # Read Excel file
                df = pd.read_excel(temp_path, engine='openpyxl')
                
                # Process the Excel file
                student_data, plot_rel_path = process_excel(df, filename=file.filename)
                
                # Remove the temporary file
                os.remove(temp_path)
                
                # Convert to list of dictionaries with correct keys
                formatted_students = []
                for student in student_data:
                    # Map the column names to match what's expected in the template
                    formatted_student = {
                        'Student Number': student.get('Student No', ''),
                        'Average English': student.get('Average English', '0.0'),
                        'Average Filipino': student.get('Average Filipino', '0.0'),
                        'Average Math': student.get('Average Math', '0.0'),
                        'Average Science': student.get('Average Science', '0.0'),
                        'Average ESP': student.get('Average ESP', '0.0'),
                        'Average ICF': student.get('Average ICT', '0.0'),  # ICF in template vs ICT in code
                        'Average TVL': student.get('Average TVL', student.get('Average TLE', student.get('Average TVE', '0.0'))),
                        'Average AP': student.get('Average AP', '0.0'),
                        'Average Mapeh': student.get('Average Mapeh', '0.0'),
                        'Average Tec. Drawing': student.get('Average Tec Drawing', '0.0'),
                        'Recommended Strand': student.get('Predicted Strand', 'N/A'),
                        'Eligible Strands': ', '.join(student.get('Eligible Strands', [])),
                        'Strand Scores': ', '.join([f"{k.replace(' Score', '')}: {v}%" for k, v in student.items() if 'Score' in k])
                    }
                    formatted_students.append(formatted_student)
                
                # Determine best strand overall (most frequent)
                strand_counts = {}
                for student in student_data:
                    strand = student['Predicted Strand']
                    strand_counts[strand] = strand_counts.get(strand, 0) + 1
                
                best_strand = max(strand_counts.items(), key=lambda x: x[1])[0] if strand_counts else "N/A"
                
                # Get the full path to the image file
                plot_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', plot_rel_path)
                
                # Read the image and convert to base64
                with open(plot_full_path, 'rb') as f:
                    img_data = f.read()
                graph_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Store transaction in the database
                user_id = session.get('user_id')
                if user_id:
                    try:
                        conn = get_db_conn()
                        cur = conn.cursor()
                        
                        # Store the strand data as JSON
                        strand_data = json.dumps({
                            'student_data': student_data,
                            'plot_path': plot_rel_path
                        })
                        
                        cur.execute(
                            """INSERT INTO transaction_logs 
                               (user_id, filename, num_students, best_strand, strand_data) 
                               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                            (user_id, file.filename, len(student_data), best_strand, strand_data)
                        )
                        
                        log_id = cur.fetchone()[0]
                        conn.commit()
                        cur.close()
                        release_db_conn(conn)
                        
                        print(f"Transaction recorded with ID: {log_id}")
                    except Exception as e:
                        print(f"Error recording transaction: {e}")
                
                print(f"Rendering template with graph_url (length: {len(graph_base64)}), {len(formatted_students)} students, best_strand: {best_strand}")
                return render_template('index.html', 
                                       graph_url=graph_base64,
                                       student_data=formatted_students,
                                       best_strand=best_strand)
            except Exception as e:
                import traceback
                traceback.print_exc()
                flash(f"Error processing file: {str(e)}")
                return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/dash')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    destroy_session()
    return redirect(url_for('login'))

@app.route('/api/predict_grades', methods=['POST'])
@login_required
def predict_grades():
    try:
        data = request.get_json()
        student_grades = pd.DataFrame(data['grades'])
        
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
            model = LogisticRegression()
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
@login_required
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

@app.route('/transaction_logs')
@login_required
def transaction_logs():
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        # Get search parameter
        search_text = request.args.get('search_text', '')

        # Build the query
        query = """
            SELECT 
                tl.id, 
                tl.filename, 
                tl.num_students, 
                tl.best_strand, 
                tl.created_at
            FROM 
                transaction_logs tl
            WHERE 1=1
        """
        params = []

        if search_text:
            query += """ AND (
                tl.filename ILIKE %s OR 
                tl.best_strand ILIKE %s
            )"""
            search_pattern = f"%{search_text}%"
            params.extend([search_pattern, search_pattern])

        query += " ORDER BY tl.created_at DESC"

        cur.execute(query, params)
        logs = cur.fetchall()
        
        return render_template('transaction_logs.html', logs=logs)
    except Exception as e:
        flash(f"Error retrieving logs: {str(e)}", 'danger')
        return render_template('transaction_logs.html', logs=[])
    finally:
        cur.close()
        release_db_conn(conn)

@app.route('/view_transaction/<int:log_id>')
@login_required
def view_transaction(log_id):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Get the transaction details
        cur.execute("""
            SELECT 
                tl.id, 
                tl.filename, 
                tl.num_students, 
                tl.best_strand, 
                tl.created_at 
            FROM 
                transaction_logs tl
            WHERE 
                tl.id = %s
        """, (log_id,))
        
        transaction = cur.fetchone()
        
        if not transaction:
            flash("Transaction log not found", "danger")
            return redirect(url_for('transaction_logs'))
        
        cur.close()
        release_db_conn(conn)
        
        # In a real implementation, you might retrieve the saved Excel data here
        # For now, we'll just return the transaction details
        
        return render_template('transaction_detail.html', transaction=transaction)
    except Exception as e:
        flash(f"Error retrieving transaction details: {e}", "danger")
        return redirect(url_for('transaction_logs'))

@app.route('/get_strand_distribution/<int:log_id>')
@login_required
def get_strand_distribution(log_id):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Get the transaction details
        cur.execute("""
            SELECT best_strand, strand_data
            FROM transaction_logs
            WHERE id = %s
        """, (log_id,))
        
        transaction = cur.fetchone()
        
        if not transaction:
            return jsonify({'success': False, 'error': 'Transaction not found'})
        
        # If we have stored strand_data as JSON, use it
        if transaction[1]:
            try:
                strand_data = json.loads(transaction[1])
                distribution = strand_data
            except:
                # If there's an error parsing JSON, create simulated data
                distribution = simulate_strand_distribution(transaction[0])
        else:
            # Otherwise simulate some distribution data based on the best strand
            distribution = simulate_strand_distribution(transaction[0])
        
        cur.close()
        release_db_conn(conn)
        
        return jsonify({
            'success': True, 
            'distribution': distribution
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def simulate_strand_distribution(best_strand):
    """Create simulated strand distribution data for visualization purposes"""
    # List of all possible strands
    strands = ['STEM', 'ABM', 'HUMSS', 'GAS', 'TVL']
    
    # Create a distribution where the best strand has more students
    distribution = {}
    
    # Give the best strand a higher count
    for strand in strands:
        if strand == best_strand:
            distribution[strand] = random.randint(5, 10)  # Higher count for best strand
        else:
            distribution[strand] = random.randint(0, 4)   # Lower counts for other strands
    
    return distribution

@app.route('/export_transaction_excel/<int:log_id>')
@login_required
def export_transaction_excel(log_id):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Get the transaction details
        cur.execute("""
            SELECT 
                tl.id, 
                tl.filename, 
                tl.num_students, 
                tl.best_strand, 
                tl.created_at 
            FROM 
                transaction_logs tl
            WHERE 
                tl.id = %s
        """, (log_id,))
        
        transaction = cur.fetchone()
        
        if not transaction:
            flash("Transaction log not found", "danger")
            return redirect(url_for('transaction_logs'))
        
        # Create a new Excel file
        output = BytesIO()
        workbook = xlsxwriter.Workbook(output)
        worksheet = workbook.add_worksheet()
        
        # Add headers
        headers = ['Transaction ID', 'Filename', 'Number of Students', 'Best Strand', 'Date Created']
        for col, header in enumerate(headers):
            worksheet.write(0, col, header)
        
        # Add transaction data
        for col, value in enumerate(transaction):
            worksheet.write(1, col, str(value))
            
        # Get strand distribution for a second sheet
        distribution = simulate_strand_distribution(transaction[3])
        
        # Create another sheet for strand distribution
        dist_worksheet = workbook.add_worksheet("Strand Distribution")
        dist_worksheet.write(0, 0, "Strand")
        dist_worksheet.write(0, 1, "Number of Students")
        
        # Write strand distribution data
        for i, (strand, count) in enumerate(distribution.items()):
            dist_worksheet.write(i+1, 0, strand)
            dist_worksheet.write(i+1, 1, count)
            
        # Create a chart for strand distribution
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Students per Strand',
            'categories': ['Strand Distribution', 1, 0, len(distribution), 0],
            'values': ['Strand Distribution', 1, 1, len(distribution), 1],
        })
        
        chart.set_title({'name': 'Strand Distribution'})
        chart.set_x_axis({'name': 'Strand'})
        chart.set_y_axis({'name': 'Number of Students'})
        
        dist_worksheet.insert_chart('D1', chart)
        
        workbook.close()
        
        # Set headers for file download
        output.seek(0)
        
        cur.close()
        release_db_conn(conn)
        
        return send_file(
            output,
            as_attachment=True,
            download_name=f"transaction_{log_id}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        flash(f"Error exporting transaction: {e}", "danger")
        return redirect(url_for('view_transaction', log_id=log_id))

@app.route('/delete_transaction/<int:log_id>', methods=['POST'])
@login_required
def delete_transaction(log_id):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Check if the transaction exists
        cur.execute("SELECT id FROM transaction_logs WHERE id = %s", (log_id,))
        if not cur.fetchone():
            return jsonify({'success': False, 'error': 'Transaction not found'})
        
        # Delete the transaction
        cur.execute("DELETE FROM transaction_logs WHERE id = %s", (log_id,))
        conn.commit()
        
        cur.close()
        release_db_conn(conn)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    setup_database()
    app.run(debug=True)
