from inference_sdk import InferenceHTTPClient
import os

def getClient(api_url, api_key):
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    
    return client

def getResult(image_path, client, workspace, workflow):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    result = client.run_workflow(
        workspace_name=workspace,
        workflow_id=workflow,
        images={"image": image_path},
        use_cache=True
    )

    # Depending on workflow, result might be JSON string or base64
    # Handle decoding/parsing outside this function
    return result

