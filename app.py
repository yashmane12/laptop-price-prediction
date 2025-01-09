from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html', companies=df['Company'].unique(), types=df['TypeName'].unique(),
                           cpus=df['Cpu brand'].unique(), gpus=df['Gpu brand'].unique(), os_list=df['os'].unique())

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        company = request.form.get('Company')
        type = request.form.get('Type')
        ram = int(request.form.get('RAM'))
        weight = float(request.form.get('Weight'))
        touchscreen = 1 if request.form.get('Touchscreen') == 'Yes' else 0
        ips = 1 if request.form.get('IPS') == 'Yes' else 0
        screen_size = float(request.form.get('ScreenSize'))
        resolution = request.form.get('Resolution')
        cpu = request.form.get('CPU')
        hdd = int(request.form.get('HDD'))
        ssd = int(request.form.get('SSD'))
        gpu = request.form.get('GPU')
        os = request.form.get('OS')

        # Process resolution and screen size
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

        # Create input query for prediction
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        # Predict the price
        predicted_price = np.exp(pipe.predict(query)[0])

        return render_template('index.html', prediction_text=f"The predicted price of this configuration is {int(predicted_price)}",
                               companies=df['Company'].unique(), types=df['TypeName'].unique(),
                               cpus=df['Cpu brand'].unique(), gpus=df['Gpu brand'].unique(), os_list=df['os'].unique())
    except Exception as e:
        return render_template('index.html', error_text=f"Error: {str(e)}", 
                               companies=df['Company'].unique(), types=df['TypeName'].unique(),
                               cpus=df['Cpu brand'].unique(), gpus=df['Gpu brand'].unique(), os_list=df['os'].unique())

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
