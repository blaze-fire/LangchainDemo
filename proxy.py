from flask import Flask, request, Response
import requests
import os
import getpass
import json

app = Flask(__name__)

MLFLOW_SERVER = "http://localhost:5000"

def get_current_user():
    """Get current Windows user"""
    return getpass.getuser()

def get_experiment_tags(experiment_id):
    """Get tags for a specific experiment"""
    try:
        response = requests.get(f"{MLFLOW_SERVER}/api/2.0/mlflow/experiments/get", 
                              params={"experiment_id": experiment_id})
        if response.status_code == 200:
            data = response.json()
            return data.get('experiment', {}).get('tags', {})
    except Exception as e:
        print(f"Error getting experiment tags: {e}")
    return {}

def filter_experiments_response(response_data, user_id):
    """Filter experiments to show only user's experiments based on tags"""
    if 'experiments' in response_data:
        filtered_experiments = []
        for exp in response_data['experiments']:
            experiment_id = exp.get('experiment_id')
            tags = exp.get('tags', {})
            
            # Check if experiment belongs to user based on tags
            # You can customize these tag keys based on your setup
            if (tags.get('user_id') == user_id or 
                tags.get('owner') == user_id or 
                tags.get('created_by') == user_id or
                exp.get('name', '').startswith(f"{user_id}_")):  # Fallback to name prefix
                filtered_experiments.append(exp)
        
        response_data['experiments'] = filtered_experiments
    return response_data

def filter_runs_response(response_data, user_id):
    """Filter runs to show only user's runs"""
    if 'runs' in response_data:
        filtered_runs = []
        for run in response_data['runs']:
            run_tags = run.get('data', {}).get('tags', {})
            
            # Check if run belongs to user
            if (run_tags.get('user_id') == user_id or 
                run_tags.get('owner') == user_id or
                run_tags.get('created_by') == user_id):
                filtered_runs.append(run)
        
        response_data['runs'] = filtered_runs
    return response_data

@app.route('/api/2.0/mlflow/experiments/list', methods=['GET'])
def list_experiments():
    """Filter experiments list for current user"""
    user_id = get_current_user()
    
    # Forward request to MLflow server
    response = requests.get(f"{MLFLOW_SERVER}/api/2.0/mlflow/experiments/list", 
                          params=request.args)
    
    if response.status_code == 200:
        data = response.json()
        filtered_data = filter_experiments_response(data, user_id)
        return filtered_data
    
    return Response(response.content, status=response.status_code, 
                   content_type=response.headers.get('content-type'))

@app.route('/api/2.0/mlflow/experiments/get', methods=['GET'])
def get_experiment():
    """Check if user can access specific experiment"""
    user_id = get_current_user()
    experiment_id = request.args.get('experiment_id')
    
    # Forward request to MLflow server
    response = requests.get(f"{MLFLOW_SERVER}/api/2.0/mlflow/experiments/get", 
                          params=request.args)
    
    if response.status_code == 200:
        data = response.json()
        experiment = data.get('experiment', {})
        tags = experiment.get('tags', {})
        
        # Check if user owns this experiment
        if (tags.get('user_id') == user_id or 
            tags.get('owner') == user_id or 
            tags.get('created_by') == user_id or
            experiment.get('name', '').startswith(f"{user_id}_")):
            return data
        else:
            # Return 403 Forbidden if user doesn't own the experiment
            return Response(json.dumps({"error": "Access denied"}), 
                          status=403, content_type='application/json')
    
    return Response(response.content, status=response.status_code, 
                   content_type=response.headers.get('content-type'))

@app.route('/api/2.0/mlflow/runs/search', methods=['POST'])
def search_runs():
    """Filter runs based on user ownership"""
    user_id = get_current_user()
    
    # Forward request to MLflow server
    response = requests.post(f"{MLFLOW_SERVER}/api/2.0/mlflow/runs/search",
                           headers=dict(request.headers),
                           json=request.get_json(),
                           params=request.args)
    
    if response.status_code == 200:
        data = response.json()
        filtered_data = filter_runs_response(data, user_id)
        return filtered_data
    
    return Response(response.content, status=response.status_code, 
                   content_type=response.headers.get('content-type'))

@app.route('/api/2.0/mlflow/runs/get', methods=['GET'])
def get_run():
    """Check if user can access specific run"""
    user_id = get_current_user()
    run_id = request.args.get('run_id')
    
    # Forward request to MLflow server
    response = requests.get(f"{MLFLOW_SERVER}/api/2.0/mlflow/runs/get", 
                          params=request.args)
    
    if response.status_code == 200:
        data = response.json()
        run = data.get('run', {})
        run_tags = run.get('data', {}).get('tags', {})
        
        # Check if user owns this run
        if (run_tags.get('user_id') == user_id or 
            run_tags.get('owner') == user_id or 
            run_tags.get('created_by') == user_id):
            return data
        else:
            return Response(json.dumps({"error": "Access denied"}), 
                          status=403, content_type='application/json')
    
    return Response(response.content, status=response.status_code, 
                   content_type=response.headers.get('content-type'))

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    """Proxy all other requests to MLflow server"""
    user_id = get_current_user()
    
    # Add user context to headers
    headers = dict(request.headers)
    headers['X-User-ID'] = user_id
    
    # Forward request
    response = requests.request(
        method=request.method,
        url=f"{MLFLOW_SERVER}/{path}",
        headers=headers,
        params=request.args,
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False
    )
    
    # Return response
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in response.raw.headers.items()
               if name.lower() not in excluded_headers]
    
    return Response(response.content, response.status_code, headers)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)