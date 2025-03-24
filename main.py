from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Save to temp file
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(temp_path)
                
                # Read Excel file
                df = pd.read_excel(temp_path, engine='openpyxl')
                
                # Process the Excel file
                plot_url, best_strand, student_data, strands, error = process_excel(df, filename=file.filename)
                
                if error:
                    flash(f"Error processing file: {error}")
                    return redirect(url_for('index'))
                
                # Remove the temporary file
                os.remove(temp_path)
                
                return render_template('index.html', 
                                     graph_url=plot_url, 
                                     best_strand=best_strand, 
                                     student_data=student_data, 
                                     strands=strands)
            except Exception as e:
                flash(f"Error processing file: {str(e)}")
                return redirect(url_for('index'))
    
    return redirect(url_for('index'))

def process_excel(df, filename="Excel Upload"):
    try:
        # Get the column indices for each subject
        grade_columns = {
            'Grade 7': {
                'English': 1,
                'Filipino': 2,
                'Math': 3,
                'ESP': 4,
                'ICF': 5,
                'TVE': 6,
                'AP': 7,
                'Mapeh': 8,
                'Tec. Drawing': 9,
                'Science': 10
            },
            'Grade 8': {
                'English': 12,
                'Filipino': 13,
                'Math': 14,
                'ESP': 15,
                'ICF': 16,
                'TVE': 17,
                'AP': 18,
                'Mapeh': 19,
                'Tec. Drawing': 20,
                'Science': 21
            },
            'Grade 9': {
                'English': 22,
                'Filipino': 23,
                'Math': 24,
                'ESP': 25,
                'ICF': 26,
                'TVE': 27,
                'AP': 28,
                'Mapeh': 29,
                'Tec. Drawing': 30,
                'Science': 31
            },
            'Grade 10': {
                'English': 32,
                'Filipino': 33,
                'Math': 34,
                'ESP': 35,
                'ICF': 36,
                'TVE': 37,
                'AP': 38,
                'Mapeh': 39,
                'Tec. Drawing': 40,
                'Science': 41
            }
        }

        # Create a clean dataset with student numbers and grades
        student_data = []
        
        # Skip the first row (header)
        for idx, row in df.iloc[1:].iterrows():
            # Skip if student number is empty or NaN
            if pd.isna(row.iloc[0]) or str(row.iloc[0]).strip() == '':
                continue
                
            student_info = {
                'Student Number': row.iloc[0],  # First column contains student number
            }
            
            # Add grades for each grade level
            for grade, subjects in grade_columns.items():
                student_info[grade] = {}
                for subject, col_idx in subjects.items():
                    try:
                        grade_value = float(row.iloc[col_idx])
                        if pd.isna(grade_value):
                            grade_value = 0.0  # Replace NaN with 0
                        student_info[grade][subject] = grade_value
                        
                        # Also store the average grades for each subject
                        if 'Average Grades' not in student_info:
                            student_info['Average Grades'] = {}
                        if subject not in student_info['Average Grades']:
                            student_info['Average Grades'][subject] = []
                        student_info['Average Grades'][subject].append(grade_value)
                    except (ValueError, IndexError):
                        print(f"Warning: Could not process grade for {subject} in {grade} for student {row.iloc[0]}")
                        continue
            
            # Calculate final average grades
            for subject in student_info['Average Grades']:
                if student_info['Average Grades'][subject]:
                    student_info['Average Grades'][subject] = sum(student_info['Average Grades'][subject]) / len(student_info['Average Grades'][subject])
                else:
                    student_info['Average Grades'][subject] = 0.0
            
            # Prepare features for strand prediction
            features = []
            for subject in ['English', 'Filipino', 'Math', 'Science', 'ESP', 'ICF', 'TVE', 'AP', 'Mapeh', 'Tec. Drawing']:
                for grade in ['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']:
                    if grade in student_info and subject in student_info[grade]:
                        features.append(student_info[grade][subject])
                    else:
                        features.append(0.0)
            
            # Create MLR model for strand prediction
            X = np.array([features])
            
            # Define strand weights for each subject
            strand_weights = {
                'STEM': {
                    'Math': 0.3,
                    'Science': 0.3,
                    'English': 0.1,
                    'ICF': 0.1,
                    'Tec. Drawing': 0.2
                },
                'HUMSS': {
                    'English': 0.3,
                    'Filipino': 0.3,
                    'AP': 0.2,
                    'ESP': 0.2
                },
                'ABM': {
                    'Math': 0.25,
                    'English': 0.25,
                    'ICF': 0.25,
                    'ESP': 0.25
                },
                'GAS': {
                    'Math': 0.2,
                    'Science': 0.2,
                    'English': 0.2,
                    'Filipino': 0.2,
                    'AP': 0.2
                },
                'TVL': {
                    'TVE': 0.4,
                    'ICF': 0.3,
                    'Tec. Drawing': 0.3
                }
            }
            
            # Calculate strand scores
            strand_scores = {}
            for strand, weights in strand_weights.items():
                score = 0
                total_weight = 0
                for subject, weight in weights.items():
                    if subject in student_info['Average Grades']:
                        score += student_info['Average Grades'][subject] * weight
                        total_weight += weight
                if total_weight > 0:
                    strand_scores[strand] = score / total_weight
                else:
                    strand_scores[strand] = 0
            
            # Get recommended strand
            recommended_strand = max(strand_scores.items(), key=lambda x: x[1])[0]
            
            # Store the scores and recommendation
            student_info['Recommended Strand'] = recommended_strand
            for strand, score in strand_scores.items():
                student_info[f'{strand} Score'] = f"{score:.1f}%"
            
            # Format the averages for display
            for subject in student_info['Average Grades']:
                student_info[f'Average {subject}'] = f"{student_info['Average Grades'][subject]:.1f}"
            
            student_data.append(student_info)
        
        # Create visualization
        plt.figure(figsize=(15, 6))
        
        # Strand Distribution Chart
        plt.subplot(1, 2, 1)
        strand_counts = {}
        for strand in strand_weights.keys():
            strand_counts[strand] = sum(1 for student in student_data if student['Recommended Strand'] == strand)
        
        plt.bar(strand_counts.keys(), strand_counts.values())
        plt.title('Distribution of Recommended Strands')
        plt.xlabel('Strand')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=45)
        
        # Average Grades by Strand Chart
        plt.subplot(1, 2, 2)
        strand_avg_grades = {strand: {'count': 0, 'total': {}} for strand in strand_weights.keys()}
        
        for student in student_data:
            strand = student['Recommended Strand']
            strand_avg_grades[strand]['count'] += 1
            for subject in ['Math', 'Science', 'English', 'Filipino', 'ICF']:
                if f'Average {subject}' in student:
                    if subject not in strand_avg_grades[strand]['total']:
                        strand_avg_grades[strand]['total'][subject] = 0
                    strand_avg_grades[strand]['total'][subject] += float(student[f'Average {subject}'].rstrip('%'))
        
        # Calculate averages
        data_for_plot = []
        labels = []
        for strand in strand_weights.keys():
            if strand_avg_grades[strand]['count'] > 0:
                averages = []
                for subject in ['Math', 'Science', 'English', 'Filipino', 'ICF']:
                    if subject in strand_avg_grades[strand]['total']:
                        avg = strand_avg_grades[strand]['total'][subject] / strand_avg_grades[strand]['count']
                        averages.append(avg)
                    else:
                        averages.append(0)
                data_for_plot.append(averages)
                labels.append(strand)
        
        x = np.arange(len(labels))
        width = 0.15
        
        for i, subject in enumerate(['Math', 'Science', 'English', 'Filipino', 'ICF']):
            plt.bar(x + i * width, [data[i] for data in data_for_plot], width, label=subject)
        
        plt.title('Average Grades by Strand')
        plt.xlabel('Strands')
        plt.ylabel('Average Grade')
        plt.xticks(x + width * 2, labels, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Get the best strand based on the number of students
        best_strand = max(strand_counts.items(), key=lambda x: x[1])[0]
        
        # Store results in the database
        conn = get_db_conn()
        cur = conn.cursor()
        
        strand_data = {
            'strand_counts': strand_counts,
            'strand_avg_grades': strand_avg_grades,
            'student_data': [
                {
                    'student_number': student['Student Number'],
                    'recommended_strand': student['Recommended Strand'],
                    'scores': {
                        strand: student[f'{strand} Score']
                        for strand in strand_weights.keys()
                    }
                }
                for student in student_data
            ]
        }
        
        cur.execute("""
            INSERT INTO transaction_logs 
            (user_id, filename, num_students, best_strand, strand_data, created_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (
            session['user_id'],
            filename,
            len(student_data),
            best_strand,
            json.dumps(strand_data)
        ))
        
        log_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        release_db_conn(conn)
        
        return plot_url, best_strand, student_data, strand_weights.keys(), None
        
    except Exception as e:
        print(f"Error in process_excel: {str(e)}")
        return None, None, None, None, str(e)

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
