#!/usr/bin/env python3
"""
Extract default temperature settings from Ollama models.
Reads models from examples/models.txt and uses ollama show command to get parameters.
"""

import subprocess
import sys
from pathlib import Path


def get_model_temperature(model_name: str) -> dict[str, str | None]:
    """Get temperature and other parameters from a model's Modelfile.
    
    Args:
        model_name: Name of the model to query.
        
    Returns:
        Dictionary with parameter information.
    """
    try:
        # Run ollama show command to get modelfile
        result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True,
            text=True,
            check=True
        )
        
        modelfile_content = result.stdout
        
        # Parse parameters from modelfile
        parameters = {
            'temperature': None,
            'top_p': None,
            'top_k': None,
            'repeat_penalty': None,
            'num_ctx': None,
        }
        
        for line in modelfile_content.split('\n'):
            line = line.strip()
            if line.startswith('PARAMETER '):
                # Format: PARAMETER parameter_name value
                parts = line.split()
                if len(parts) >= 3:
                    param_name = parts[1]
                    param_value = parts[2]
                    
                    if param_name in parameters:
                        parameters[param_name] = param_value
        
        return {
            'model': model_name,
            'status': 'success',
            'temperature': parameters.get('temperature'),
            'top_p': parameters.get('top_p'),
            'top_k': parameters.get('top_k'),
            'repeat_penalty': parameters.get('repeat_penalty'),
            'num_ctx': parameters.get('num_ctx'),
            'raw_output': modelfile_content,
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'model': model_name,
            'status': 'error',
            'error': f"Command failed: {e}",
            'stderr': e.stderr,
        }
    except FileNotFoundError:
        return {
            'model': model_name,
            'status': 'error',
            'error': "ollama command not found",
        }
    except Exception as e:
        return {
            'model': model_name,
            'status': 'error',
            'error': f"Unexpected error: {e}",
        }


def load_models_from_file(models_file: Path) -> list[str]:
    """Load model names from file, ignoring comments and empty lines.
    
    Args:
        models_file: Path to the models file.
        
    Returns:
        List of model names.
    """
    if not models_file.exists():
        print(f"Error: Models file not found: {models_file}")
        sys.exit(1)
    
    models = []
    try:
        with open(models_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    models.append(line)
    except Exception as e:
        print(f"Error reading models file: {e}")
        sys.exit(1)
    
    return models


def main():
    """Main function."""
    # Path to models file
    models_file = Path("examples/models.txt")
    
    # Load models
    models = load_models_from_file(models_file)
    
    if not models:
        print("No models found in examples/models.txt")
        sys.exit(1)
    
    print(f"Extracting parameters for {len(models)} models...\n")
    
    # Results storage
    results = []
    
    # Process each model
    for model in models:
        print(f"Processing: {model}")
        result = get_model_temperature(model)
        results.append(result)
        
        if result['status'] == 'success':
            temp = result.get('temperature', 'Not set')
            top_p = result.get('top_p', 'Not set')
            top_k = result.get('top_k', 'Not set')
            print(f"  Temperature: {temp}")
            print(f"  Top-p: {top_p}")
            print(f"  Top-k: {top_k}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        print()
    
    # Summary table
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<40} {'Temperature':<12} {'Top-p':<8} {'Top-k':<8} {'Status'}")
    print("-" * 80)
    
    for result in results:
        model = result['model']
        if len(model) > 39:
            model = model[:36] + "..."
        
        if result['status'] == 'success':
            temp = result.get('temperature') or 'N/A'
            top_p = result.get('top_p') or 'N/A'
            top_k = result.get('top_k') or 'N/A'
            status = "OK"
        else:
            temp = top_p = top_k = "-"
            status = "ERROR"
        
        print(f"{model:<40} {temp:<12} {top_p:<8} {top_k:<8} {status}")
    
    # Save detailed results
    output_file = Path("model_parameters.txt")
    with open(output_file, 'w') as f:
        f.write("Model Parameter Extraction Results\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Status: {result['status']}\n")
            
            if result['status'] == 'success':
                f.write(f"Temperature: {result.get('temperature', 'Not set')}\n")
                f.write(f"Top-p: {result.get('top_p', 'Not set')}\n")
                f.write(f"Top-k: {result.get('top_k', 'Not set')}\n")
                f.write(f"Repeat penalty: {result.get('repeat_penalty', 'Not set')}\n")
                f.write(f"Context length: {result.get('num_ctx', 'Not set')}\n")
                f.write("\nRaw Modelfile:\n")
                f.write("-" * 30 + "\n")
                f.write(result.get('raw_output', 'No output'))
                f.write("\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                if 'stderr' in result:
                    f.write(f"Stderr: {result['stderr']}\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()