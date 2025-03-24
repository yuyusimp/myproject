-- Drop tables if they exist
DROP TABLE IF EXISTS transaction_logs;
DROP TABLE IF EXISTS users;

-- Create users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(200) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create transaction_logs table
CREATE TABLE transaction_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    num_students INTEGER NOT NULL,
    best_strand VARCHAR(50) NOT NULL,
    accuracy_score DECIMAL(5,2),
    strand_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_transaction_logs_user_id ON transaction_logs(user_id);
CREATE INDEX idx_transaction_logs_created_at ON transaction_logs(created_at);

-- Insert sample admin user (password: admin123)
INSERT INTO users (username, email, password_hash, role) VALUES 
('admin', 'admin@school.edu', 'pbkdf2:sha256:600000$7qzUBysRJSEtKgEa$c1d7a303e516693528d24c3ceb2694bd4c95af48e7e6e9b1d425c91c75e40f37', 'admin');

-- Insert sample regular users (password: password123)
INSERT INTO users (username, email, password_hash) VALUES 
('teacher1', 'teacher1@school.edu', 'pbkdf2:sha256:600000$X8nTxXL6pE3RJPRM$6d028a76b46c0f0e64b0c31e7cbf3471b3b4287c9e8c43c5440c253c29d9f478'),
('teacher2', 'teacher2@school.edu', 'pbkdf2:sha256:600000$X8nTxXL6pE3RJPRM$6d028a76b46c0f0e64b0c31e7cbf3471b3b4287c9e8c43c5440c253c29d9f478');

-- Insert sample transaction logs
INSERT INTO transaction_logs (user_id, filename, num_students, best_strand, accuracy_score, strand_data) VALUES 
(1, 'batch1_2024.xlsx', 150, 'STEM', 92.5, '{"STEM": 45, "ABM": 30, "HUMSS": 25, "GAS": 30, "TVL": 20}'),
(1, 'batch2_2024.xlsx', 120, 'ABM', 88.7, '{"STEM": 25, "ABM": 35, "HUMSS": 20, "GAS": 25, "TVL": 15}'),
(2, 'class1A_2024.xlsx', 45, 'HUMSS', 90.2, '{"STEM": 10, "ABM": 8, "HUMSS": 15, "GAS": 7, "TVL": 5}'),
(3, 'class2B_2024.xlsx', 42, 'STEM', 91.8, '{"STEM": 15, "ABM": 8, "HUMSS": 7, "GAS": 7, "TVL": 5}');

-- Comments on database schema:
/*
1. Users Table:
   - Stores user authentication and profile information
   - Uses SERIAL for auto-incrementing IDs
   - Includes email for communication
   - Role field for access control (admin/user)
   - Timestamps for auditing

2. Transaction Logs Table:
   - Records all file uploads and analysis results
   - Links to users table for tracking who performed the analysis
   - Stores strand distribution data as JSON in TEXT field
   - Includes accuracy score for model performance tracking
   - ON DELETE CASCADE ensures referential integrity

3. Indexes:
   - Added for frequently queried columns
   - Improves search and sort performance
   - Covers common filtering scenarios

4. Sample Data:
   - Includes admin and regular user accounts
   - Demonstrates various strand distributions
   - Shows different accuracy scores
   - Uses realistic filenames and student counts
*/
