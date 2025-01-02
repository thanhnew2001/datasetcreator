from flask import Flask, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define a simple route
@app.route('/')
def home():
    return jsonify({"message": "Hello, Flask!"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
