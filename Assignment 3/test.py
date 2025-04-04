

import unittest
import requests
import subprocess
import time
from score import score
import joblib

class TestScoreFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the best model saved during experiments
        cls.model = joblib.load("/content/best_model.pkl")

    def test_smoke(self):
        """Check if the function runs without crashing."""
        text = "Hello, this is a test message."
        prediction, propensity = score(text, self.model, threshold=0.5)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)

    def test_output_format(self):
        """Check if output types are as expected."""
        text = "Sample text."
        prediction, propensity = score(text, self.model, threshold=0.5)
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)

    def test_prediction_values(self):
        """Ensure prediction is always 0 or 1."""
        text = "Another test message."
        prediction, _ = score(text, self.model, threshold=0.5)
        self.assertIn(prediction, [0, 1])

    def test_propensity_range(self):
        """Ensure propensity score is between 0 and 1."""
        text = "Check propensity range."
        _, propensity = score(text, self.model, threshold=0.5)
        self.assertGreaterEqual(propensity, 0.0)
        self.assertLessEqual(propensity, 1.0)

    def test_threshold_zero(self):
        """With threshold 0, prediction should always be 1."""
        text = "Threshold zero test."
        prediction, _ = score(text, self.model, threshold=0)
        self.assertEqual(prediction, 1)

    def test_threshold_one(self):
        """With threshold 1, prediction should always be 0."""
        text = "Threshold one test."
        prediction, _ = score(text, self.model, threshold=1)
        self.assertEqual(prediction, 0)

    def test_obvious_spam(self):
        """Test an obvious spam input, expecting prediction 1."""
        text = "Congratulations! You won a free lottery. Click here!"
        prediction, _ = score(text, self.model, threshold=0.5)
        self.assertEqual(prediction, 1)

    def test_obvious_non_spam(self):
        """Test an obvious non-spam input, expecting prediction 0."""
        text = "Hello John, let's meet for coffee tomorrow."
        prediction, _ = score(text, self.model, threshold=0.5)
        self.assertEqual(prediction, 0)

class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the Flask server in the background."""
        cls.process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(10)  # Give the server time to start

    @classmethod
    def tearDownClass(cls):
        """Terminate the Flask server."""
        cls.process.terminate()
        cls.process.wait()

    def test_flask(self):
        """Integration test for Flask API."""
        url = "http://127.0.0.1:5000/score"
        data = {"text": "Free money! Claim now!", "threshold": 0.5}
        response = requests.post(url, json=data)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("prediction", result)
        self.assertIn("propensity", result)

if __name__ == "__main__":
    unittest.main()
