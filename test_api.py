#!/usr/bin/env python3
"""
Test script for the improved Fake News Detector API
"""

import requests
import json

def test_api():
    print("Testing Fake News Detector API...")

    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get('http://localhost:5000/health', timeout=5)
        print(f"   Health check: {response.status_code}")

        # Test detection endpoint
        print("\n2. Testing detection endpoint...")
        test_data = {'text': 'Breaking news: Scientists discover new planet in our solar system'}
        response = requests.post('http://localhost:5000/detect',
                               json=test_data,
                               headers={'Content-Type': 'application/json'},
                               timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("   Detection successful:")
            print(f"   Result: {result['result']}")
            print(f"   Fake %: {result['fake_percentage']}")
            print(f"   Real %: {result['real_percentage']}")
            print(f"   Detection ID: {result['detection_id']}")
        else:
            print(f"   Detection failed: {response.status_code} - {response.text}")

        # Test input validation
        print("\n3. Testing input validation...")
        bad_data = {'text': 'short'}
        response = requests.post('http://localhost:5000/detect',
                               json=bad_data,
                               headers={'Content-Type': 'application/json'},
                               timeout=10)
        print(f"   Short text validation: {response.status_code}")

        # Test history endpoint
        print("\n4. Testing history endpoint...")
        response = requests.get('http://localhost:5000/history', timeout=10)
        if response.status_code == 200:
            history = response.json()
            print(f"   History retrieved: {history['total']} items")
        else:
            print(f"   History failed: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("❌ Flask app not running - start with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print("\n✅ All tests completed!")
    return True

if __name__ == '__main__':
    test_api()