#!/usr/bin/env python3

import subprocess
import json
import time
import sys
import logging
import argparse
import os
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vastai_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

class VastAIBenchmark:
    def __init__(self):
        self.instance_id = None
        self.ssh_info = None
        self.scp_info = None
        
    def run_command(self, cmd, capture_output=True, check=True):
        """Run a shell command with logging"""
        logger.info(f"Running command: {cmd}")
        try:
            if capture_output:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
                if result.stdout:
                    logger.debug(f"stdout: {result.stdout.strip()}")
                if result.stderr:
                    logger.debug(f"stderr: {result.stderr.strip()}")
                return result
            else:
                result = subprocess.run(cmd, shell=True, check=check)
                return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {cmd}")
            logger.error(f"Return code: {e.returncode}")
            if hasattr(e, 'stdout') and e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"stderr: {e.stderr}")
            raise
    
    def search_and_launch_instance(self, gpu_type="RTX_4090", region="Asia", disk_size=32.0, num_gpus=1):
        """Search for available instances and launch the first suitable one"""
        logger.info(f"Searching for available {gpu_type} instances...")
        
        # Search for available instances
        search_cmd = f'vastai search instances "gpu_name == {gpu_type} num_gpus={num_gpus}" --raw'
        search_result = self.run_command(search_cmd)
        
        try:
            available_instances = json.loads(search_result.stdout)
            
            if not available_instances:
                raise ValueError(f"No available {gpu_type} instances found")
            
            # Sort by price (ascending) to get the cheapest option first
            available_instances.sort(key=lambda x: float(x.get('dph_total', float('inf'))))
            
            # Find the first suitable instance
            selected_instance = None
            for instance in available_instances:
                # Check if it meets our requirements
                if (instance.get('disk_space', 0) >= disk_size and 
                    instance.get('machine_id') and
                    instance.get('ask_contract_id')):
                    selected_instance = instance
                    break
            
            if not selected_instance:
                raise ValueError(f"No suitable {gpu_type} instances found with {disk_size}GB+ disk space")
            
            ask_contract_id = selected_instance['ask_contract_id']
            price = selected_instance.get('dph_total', 'unknown')
            location = selected_instance.get('geolocation', 'unknown')
            
            logger.info(f"Selected instance: Contract ID {ask_contract_id}, Price: ${price}/hour, Location: {location}")
            
            # Create the selected instance
            create_cmd = f"vastai create instance {ask_contract_id} --image vastai/base-image:cuda-12.8.1-auto --disk {disk_size} --raw"
            logger.info(f"Creating instance with contract ID: {ask_contract_id}")
            result = self.run_command(create_cmd)
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse search results: {search_result.stdout}")
            raise
        
        try:
            instance_data = json.loads(result.stdout)
            
            # Handle different response formats from VastAI
            self.instance_id = (
                instance_data.get('new_contract') or 
                instance_data.get('id') or 
                instance_data.get('instance_id')
            )
            
            if not self.instance_id:
                logger.error(f"Could not find instance ID in response: {instance_data}")
                raise ValueError("Instance ID not found in response")
            
            # Ensure it's a string for consistent handling
            self.instance_id = str(self.instance_id)
            logger.info(f"Instance launched successfully. ID: {self.instance_id}")
            return self.instance_id
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {result.stdout}")
            # Try to extract instance ID from text output
            match = re.search(r'"id":\s*(\d+)', result.stdout)
            if match:
                self.instance_id = match.group(1)
                logger.info(f"Extracted instance ID from text: {self.instance_id}")
                return self.instance_id
            raise
    
    def wait_for_ready(self, timeout=1800):  # 30 minutes timeout
        """Wait for the instance to be ready"""
        logger.info(f"Waiting for instance {self.instance_id} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.run_command("vastai show instances --raw")
                
                # Parse JSON response to find our instance
                instances = json.loads(result.stdout)
                
                # Find our instance in the list
                for instance in instances:
                    if str(instance.get('id')) == str(self.instance_id):
                        status = instance.get('actual_status', 'unknown')
                        logger.info(f"Instance {self.instance_id} status: {status}")
                        
                        # Check if instance is ready (running state)
                        if status.lower() in ['running', 'ready']:
                            logger.info("Instance is ready!")
                            return True
                        elif status.lower() in ['failed', 'error', 'stopped']:
                            raise RuntimeError(f"Instance {self.instance_id} failed with status: {status}")
                        
                        break
                else:
                    logger.warning(f"Instance {self.instance_id} not found in instances list")
                
                logger.info("Instance not ready yet, waiting 30 seconds...")
                time.sleep(5)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                time.sleep(5)
            except Exception as e:
                logger.warning(f"Error checking instance status: {e}")
                time.sleep(5)
        
        raise TimeoutError(f"Instance {self.instance_id} did not become ready within {timeout} seconds")
    
    def get_connection_info(self):
        """Get SSH and SCP connection information"""
        logger.info("Getting connection information...")
        
        # Get SSH URL
        ssh_result = self.run_command(f"vastai ssh-url {self.instance_id}")
        self.ssh_info = ssh_result.stdout.strip()
        logger.info(f"SSH info: {self.ssh_info}")
        
        # Get SCP URL  
        scp_result = self.run_command(f"vastai scp-url {self.instance_id}")
        self.scp_info = scp_result.stdout.strip()
        logger.info(f"SCP info: {self.scp_info}")
        
        return self.ssh_info, self.scp_info
    
    def copy_files(self):
        """Copy required files to the remote instance"""
        logger.info("Copying files to remote instance...")
        
        files_to_copy = ['patch.diff', 'setup_script.sh']
        
        for file_path in files_to_copy:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
            
            logger.info(f"Copying {file_path}...")
            
            # Parse SCP URL format: scp://user@host:port
            if self.scp_info.startswith('scp://'):
                # Extract user@host:port from scp://user@host:port
                connection_part = self.scp_info[6:]  # Remove 'scp://'
                if '@' in connection_part and ':' in connection_part:
                    user_host, port = connection_part.rsplit(':', 1)
                    cmd = f"scp -P {port} -o StrictHostKeyChecking=no {file_path} {user_host}:~/"
                else:
                    raise ValueError(f"Unexpected SCP URL format: {self.scp_info}")
            else:
                # Fallback for other formats
                raise ValueError(f"Unsupported SCP URL format: {self.scp_info}")
            
            self.run_command(cmd)
            logger.info(f"Successfully copied {file_path}")
    
    def run_setup_script(self):
        """Run the setup script on the remote instance"""
        logger.info("Running setup script on remote instance...")
        
        # Parse SSH URL format: ssh://user@host:port
        if self.ssh_info.startswith('ssh://'):
            # Extract user@host:port from ssh://user@host:port
            connection_part = self.ssh_info[6:]  # Remove 'ssh://'
            if '@' in connection_part and ':' in connection_part:
                user_host, port = connection_part.rsplit(':', 1)
            else:
                raise ValueError(f"Unexpected SSH URL format: {self.ssh_info}")
        else:
            raise ValueError(f"Unsupported SSH URL format: {self.ssh_info}")
        
        # Make the script executable and run it
        setup_commands = [
            "chmod +x ~/setup_script.sh",
            "cd ~ && ./setup_script.sh 2>&1 | tee setup_output.log"
        ]
        
        for cmd in setup_commands:
            full_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no {user_host} '{cmd}'"
            logger.info(f"Executing: {cmd}")
            self.run_command(full_cmd, capture_output=False)
    
    def get_results(self):
        """Get the benchmark results from the remote instance"""
        logger.info("Retrieving benchmark results...")
        
        # Parse connection details from URL format
        if self.ssh_info.startswith('ssh://'):
            connection_part = self.ssh_info[6:]  # Remove 'ssh://'
            if '@' in connection_part and ':' in connection_part:
                user_host, port = connection_part.rsplit(':', 1)
            else:
                raise ValueError(f"Unexpected SSH URL format: {self.ssh_info}")
        else:
            raise ValueError(f"Unsupported SSH URL format: {self.ssh_info}")
        
        # Try to copy results file back using SCP
        results_cmd = f"scp -P {port} -o StrictHostKeyChecking=no {user_host}:~/llama.cpp/results.out.txt ./vastai_results.txt"
        try:
            self.run_command(results_cmd)
            logger.info("Results downloaded to vastai_results.txt")
            
            # Display results
            with open('vastai_results.txt', 'r') as f:
                results = f.read()
                logger.info("Benchmark Results:")
                print("\n" + "="*50)
                print("BENCHMARK RESULTS")
                print("="*50)
                print(results)
                print("="*50)
                
        except Exception as e:
            logger.warning(f"Could not download results file: {e}")
            # Try to get results via SSH
            results_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no {user_host} 'cat ~/llama.cpp/results.out.txt'"
            try:
                result = self.run_command(results_cmd)
                logger.info("Benchmark Results (via SSH):")
                print("\n" + "="*50)
                print("BENCHMARK RESULTS")
                print("="*50)
                print(result.stdout)
                print("="*50)
            except Exception as e2:
                logger.error(f"Could not retrieve results: {e2}")
    
    def cleanup_instance(self):
        """Cleanup the VastAI instance"""
        if self.instance_id:
            logger.info(f"Cleaning up instance {self.instance_id}...")
            try:
                self.run_command(f"vastai destroy instance {self.instance_id}")
                logger.info("Instance destroyed successfully")
            except Exception as e:
                logger.error(f"Failed to destroy instance: {e}")
    
    def run_benchmark(self, gpu_type="RTX_4090", region="Asia", disk_size=32.0, num_gpus=1, instance_id=None, cleanup=True):
        """Run the complete benchmark workflow"""
        try:
            if instance_id:
                # Use provided instance ID
                logger.info(f"Using provided instance ID: {instance_id}")
                self.instance_id = str(instance_id)
                
                # Step 1: Wait for ready (skip creation)
                self.wait_for_ready()
            else:
                # Step 1: Search and launch instance
                self.search_and_launch_instance(gpu_type, region, disk_size, num_gpus)
                
                # Step 2: Wait for ready
                self.wait_for_ready()
            
            # Step 2: Get connection info
            self.get_connection_info()
            
            # Step 3: Copy files
            self.copy_files()
            
            # Step 4: Run setup script
            self.run_setup_script()
            
            # Step 5: Get results
            self.get_results()
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        finally:
            if cleanup and self.instance_id:
                self.cleanup_instance()

def main():
    parser = argparse.ArgumentParser(description='Run VastAI benchmark automation')
    parser.add_argument('--gpu-type', default='RTX_4090', help='GPU type to request')
    parser.add_argument('--region', default='Asia', help='Region preference')
    parser.add_argument('--disk-size', type=float, default=32.0, help='Disk size in GB')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to request')
    parser.add_argument('--instance-id', type=str, help='Use existing instance ID (skip creation)')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not destroy instance after completion')
    
    args = parser.parse_args()
    
    benchmark = VastAIBenchmark()
    
    try:
        benchmark.run_benchmark(
            gpu_type=args.gpu_type,
            region=args.region,
            disk_size=args.disk_size,
            num_gpus=args.num_gpus,
            instance_id=args.instance_id,
            cleanup=not args.no_cleanup
        )
        logger.info("Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        if benchmark.instance_id:
            logger.info("Cleaning up instance...")
            benchmark.cleanup_instance()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if benchmark.instance_id and not args.no_cleanup:
            benchmark.cleanup_instance()
        sys.exit(1)

if __name__ == "__main__":
    main()
