"""
=============================================================================
AI-DRIVEN EARLY WARNING SYSTEM FOR PREDICTING STUDENT DROPOUT
IN NIGERIAN UNIVERSITIES — LIGHTWEIGHT WEB BACKEND
=============================================================================
Author  : Senior Data Scientist & ML Engineer (EDM Specialisation)
Description : Exposes API endpoints and serves the EWS dashboard using python http.server.
"""

import os
import json
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
import pandas as pd
from urllib.parse import urlparse, parse_qs

# Adjust path to import ews_inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ews_inference import predict_dropout_risk, load_assets, predict_dropout_risk_batch

PORT = 8000

# Global caches for student dataset and predictions
_students_df = None
_students_predictions = []
_students_summary = []

class EWSRequestHandler(SimpleHTTPRequestHandler):
    """Subclassing standard HTTP request handler to serve endpoints and dashboard files."""
    
    def log_message(self, format, *args):
        # Silence default log outputs to keep stdout clean
        pass
        
    def do_OPTIONS(self):
        # Enable CORS headers for preflight request
        self.send_response(200, "OK")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # CORS headers
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                # Load student attributes dictionary
                student_data = json.loads(post_data.decode('utf-8'))
                
                # Execute inference
                result = predict_dropout_risk(student_data)
                
                # Send prediction response
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                error_response = {"error": str(e)}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

    def do_GET(self):
        # Parse path and query parameters
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        
        # CORS headers helper
        def send_json_headers():
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

        if path == '/api/metrics':
            send_json_headers()
            metrics_path = "models/metrics_summary.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                error_resp = {"error": "Metrics summary file not found. Run train_ews.py first."}
                self.wfile.write(json.dumps(error_resp).encode('utf-8'))
                
        elif path == '/api/students':
            send_json_headers()
            
            # Filtering
            filtered = _students_summary
            
            # Risk tier filter
            tier = query.get('risk_tier', ['All'])[0]
            if tier == 'High & Medium Risk':
                filtered = [s for s in filtered if s['risk_tier'] in ('High Risk', 'Medium Risk')]
            elif tier != 'All' and tier in ('High Risk', 'Medium Risk', 'Low Risk'):
                filtered = [s for s in filtered if s['risk_tier'] == tier]
                
            # Faculty filter
            faculty = query.get('faculty', ['All'])[0]
            if faculty != 'All':
                filtered = [s for s in filtered if s['faculty'].lower() == faculty.lower()]
                
            # Search filter
            search = query.get('search', [''])[0].strip().lower()
            if search:
                filtered = [s for s in filtered if search in s['student_id'].lower() or search in s['faculty'].lower()]
                
            # Sorting
            sort_by = query.get('sort_by', ['student_id'])[0]
            sort_order = query.get('sort_order', ['asc'])[0]
            reverse_sort = (sort_order == 'desc')
            
            if sort_by == 'cgpa':
                filtered = sorted(filtered, key=lambda x: x['cgpa'], reverse=reverse_sort)
            elif sort_by == 'probability':
                filtered = sorted(filtered, key=lambda x: x['risk_probability'], reverse=reverse_sort)
            else: # Default student_id sorting
                try:
                    filtered = sorted(filtered, key=lambda x: int(x['student_id'].split('-')[1]), reverse=reverse_sort)
                except Exception:
                    filtered = sorted(filtered, key=lambda x: x['student_id'], reverse=reverse_sort)
                
            # Pagination
            try:
                page = int(query.get('page', [1])[0])
                limit = int(query.get('limit', [15])[0])
            except ValueError:
                page = 1
                limit = 15
                
            total_records = len(filtered)
            start = (page - 1) * limit
            end = start + limit
            paginated = filtered[start:end]
            
            response_data = {
                "students": paginated,
                "total": total_records,
                "page": page,
                "limit": limit,
                "pages": (total_records + limit - 1) // limit
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            
        elif path == '/api/student-detail':
            send_json_headers()
            stu_id = query.get('id', [''])[0].strip().upper()
            
            try:
                # Format STU-xxxx
                idx = int(stu_id.split('-')[1]) - 1
                if _students_df is not None and 0 <= idx < len(_students_df):
                    row = _students_df.iloc[idx].to_dict()
                    
                    # Convert types to python standard types
                    for k, v in row.items():
                        if isinstance(v, (np.integer, np.int64)):
                            row[k] = int(v)
                        elif isinstance(v, (np.floating, np.float64)):
                            row[k] = float(v)
                            
                    pred = _students_predictions[idx]
                    
                    response_data = {
                        "student_id": stu_id,
                        "features": row,
                        "prediction": pred
                    }
                    self.wfile.write(json.dumps(response_data).encode('utf-8'))
                else:
                    self.wfile.write(json.dumps({"error": f"Student ID {stu_id} out of bounds"}).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({"error": f"Invalid student ID format: {str(e)}"}).encode('utf-8'))

        elif path in ('/', '/dashboard'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            dashboard_path = "dashboard.html"
            if os.path.exists(dashboard_path):
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.wfile.write(b"<h1>Dashboard file dashboard.html not found in workspace directory</h1>")
                
        elif path == '/reports/ews_evaluation_report.png':
            report_path = "reports/ews_evaluation_report.png"
            if os.path.exists(report_path):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(report_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            # Fallback to serving regular static files in local path
            super().do_GET()

def run_server():
    global _students_df, _students_predictions, _students_summary
    
    # Attempt to load EWS model assets on startup
    loaded = load_assets()
    if loaded:
        print("[+] Operational EWS model successfully pre-loaded for server.")
    else:
        print("[-] Warning: Model assets not found in 'models/'. Server will run, but API calls will fail until 'train_ews.py' is run.")
        
    # Pre-load dataset and pre-compute predictions
    try:
        csv_path = "data/Actual_nigerian_student_dropout_dataset.csv"
        if os.path.exists(csv_path) and loaded:
            print("[*] Loading student dataset and pre-computing risk profiles...")
            import time
            t0 = time.time()
            
            _students_df = pd.read_csv(csv_path)
            # Precompute batch predictions
            preds = predict_dropout_risk_batch(_students_df)
            _students_predictions = preds
            _students_summary = []
            
            for idx, row in _students_df.iterrows():
                pred = preds[idx]
                stu_id = f"STU-{idx+1:04d}"
                gender_label = "Male" if int(row["Gender"]) == 1 else "Female"
                
                _students_summary.append({
                    "student_id": stu_id,
                    "faculty": row["Faculty"],
                    "cgpa": float(row["CGPA_5point_Scale"]),
                    "year_of_study": int(row["Year_of_Study"]),
                    "gender": gender_label,
                    "risk_probability": float(pred["probability"]),
                    "risk_tier": pred["risk_tier"]
                })
            
            print(f"[+] Successfully loaded and precomputed {len(_students_summary)} student profiles in {time.time() - t0:.3f}s.")
        else:
            print("[-] Warning: Dataset CSV or model assets missing. Student Directory feature will be disabled.")
    except Exception as e:
        print(f"[-] Error loading student dataset: {e}")

    server_address = ('', PORT)
    httpd = HTTPServer(server_address, EWSRequestHandler)
    print(f"\n=========================================================================")
    print(f"  AI-Driven EWS Web Dashboard Server Running Local Instance")
    print(f"  Endpoint: http://localhost:{PORT}/")
    print(f"=========================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        print("Server shutdown complete.")

if __name__ == '__main__':
    # Need numpy to handle internal types
    import numpy as np
    run_server()
