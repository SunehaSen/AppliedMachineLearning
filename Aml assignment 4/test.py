import unittest
import requests
import subprocess
import time
from score import score
import joblib
import os
import shutil

class TestScoreFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the best model saved during experiments
        cls.model = joblib.load("best_model.pkl")  # Adjust path as needed

    def test_smoke(self):
        text = "Hello, this is a test message."
        prediction, propensity = score(text, self.model, threshold=0.5)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)

    def test_output_format(self):
        text = "Sample text."
        prediction, propensity = score(text, self.model, threshold=0.5)
        self.assertIsInstance(prediction, (bool, int))
        self.assertIsInstance(propensity, float)

    def test_prediction_values(self):
        text = "Another test message."
        prediction, _ = score(text, self.model, threshold=0.5)
        self.assertIn(prediction, [0, 1])

    def test_propensity_range(self):
        text = "Check propensity range."
        _, propensity = score(text, self.model, threshold=0.5)
        self.assertGreaterEqual(propensity, 0.0)
        self.assertLessEqual(propensity, 1.0)

    def test_threshold_zero(self):
        text = "Threshold zero test."
        prediction, _ = score(text, self.model, threshold=0)
        self.assertEqual(prediction, 1)

    def test_threshold_one(self):
        text = "Threshold one test."
        prediction, _ = score(text, self.model, threshold=1)
        self.assertEqual(prediction, 0)

    def test_obvious_spam(self):
        text = "Congratulations! You won a free lottery. Click here!"
        prediction, _ = score(text, self.model, threshold=0.5)
        self.assertEqual(prediction, 1)

    def test_obvious_non_spam(self):
        text = "Hello John, let's meet for coffee tomorrow."
        prediction, _ = score(text, self.model, threshold=0.5)
        self.assertEqual(prediction, 0)


class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(10)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait()

    def test_flask(self):
        url = "http://127.0.0.1:5000/score"
        data = {"text": "Free money! Claim now!", "threshold": 0.5}
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("prediction", result)
        self.assertIn("propensity", result)


class TestDocker(unittest.TestCase):
    def test_docker(self):
        image_name = "flask-assignment4-app"
        container_name = "flask-assignment4-container"

        try:
            # Step 1: Build the Docker image
            subprocess.run(["docker", "build", "-t", image_name, "."], check=True)

            # Step 2: Run the Docker container
            run_cmd = [
                "docker", "run", "-d", "--rm",
                "-p", "5050:5000",
                "--name", container_name,
                image_name
            ]
            subprocess.run(run_cmd, check=True)

            # Allow Flask to start up
            time.sleep(5)

            # Step 3: Send a request to the container
            url = "http://localhost:5050/score"
            data = {"text": "This is a test sentence", "threshold": 0.5}
            response = requests.post(url, json=data)
            
            # Step 4: Check the response
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn("prediction", result)
            self.assertIn("propensity", result)
            
        finally:
            # Step 5: Clean up - stop the container
            subprocess.run(["docker", "stop", container_name], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()