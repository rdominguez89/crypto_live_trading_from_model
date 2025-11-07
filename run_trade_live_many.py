import os
import time
import subprocess

models = ['l6mod','l7mod','l8mod','l15mod']
output_files = []
for model_option in models:
    if model_option[0]=='l':side='long'
    if model_option[0]=='s':side='short'
    number_model = model_option[1:].replace('mod','')
    output_files.append(f'sal_{side}_{number_model}.dat')
    #use lsof to check if it is running by getting PID and killing it

# Kill any existing processes using the output files
for output_file in output_files:
    try:
        # Use lsof to find PIDs using the output files
        result = subprocess.run(['lsof', '-t', output_file], 
                              capture_output=True, text=True)
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Killing existing process with PID {pid} using {output_file}")
                    subprocess.run(['kill', '-9', pid])
                    time.sleep(1)
    except Exception as e:
        print(f"Error checking/killing processes for {output_file}: {e}")

for k, model_option in enumerate(models):
    if model_option == 'l6mod': continue
    # Write model_option to input.txt
    with open('input.txt', 'w') as f:
        f.write(model_option + '\n')
    
    command = f'python -u trade_live.py < input.txt >> {output_files[k]} 2>&1 &'
    print(command)
    os.system(command)
    if(k<len(models)-1):time.sleep(30)

print("All processes started successfully!")