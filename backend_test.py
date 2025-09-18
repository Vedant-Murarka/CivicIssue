#!/usr/bin/env python3

import requests
import sys
import json
import time
from datetime import datetime
import os

class CivicIssueAPITester:
    def __init__(self, base_url="https://fixmycity-22.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.citizen_token = None
        self.admin_token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_user_phone = f"+1555{int(time.time()) % 1000000:06d}"
        self.admin_phone = "+19999999999"
        self.admin_password = "admin123"

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        if details and success:
            print(f"   Details: {details}")

    def make_request(self, method, endpoint, data=None, files=None, token=None, expected_status=200, form_data=False):
        """Make HTTP request with error handling"""
        url = f"{self.api_url}/{endpoint}"
        headers = {}
        
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        if not files and not form_data:
            headers['Content-Type'] = 'application/json'

        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files or form_data:
                    response = requests.post(url, data=data, files=files, headers=headers)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers)
            else:
                return False, f"Unsupported method: {method}"

            success = response.status_code == expected_status
            
            if success:
                try:
                    return True, response.json()
                except:
                    return True, response.text
            else:
                try:
                    error_detail = response.json().get('detail', response.text)
                except:
                    error_detail = response.text
                return False, f"Status {response.status_code}: {error_detail}"

        except Exception as e:
            return False, f"Request failed: {str(e)}"

    def test_admin_creation(self):
        """Test admin user creation"""
        print("\n🔧 Testing Admin Creation...")
        success, response = self.make_request('POST', 'admin/create', expected_status=200)
        self.log_test("Admin Creation", success, str(response))
        return success

    def test_user_registration(self):
        """Test user registration"""
        print("\n👤 Testing User Registration...")
        
        user_data = {
            "phone": self.test_user_phone,
            "full_name": "Test User",
            "password": "testpass123"
        }
        
        success, response = self.make_request('POST', 'auth/register', data=user_data, expected_status=200)
        
        if success and isinstance(response, dict):
            otp = response.get('otp')
            user_id = response.get('user_id')
            details = f"OTP: {otp}, User ID: {user_id}"
        else:
            details = str(response)
            
        self.log_test("User Registration", success, details)
        return success

    def test_otp_verification(self):
        """Test OTP verification"""
        print("\n🔐 Testing OTP Verification...")
        
        otp_data = {
            "phone": self.test_user_phone,
            "otp": "123456"
        }
        
        success, response = self.make_request('POST', 'auth/verify-otp', data=otp_data, expected_status=200)
        
        if success and isinstance(response, dict):
            self.citizen_token = response.get('access_token')
            user_info = response.get('user', {})
            details = f"Token received, User: {user_info.get('full_name')} ({user_info.get('role')})"
        else:
            details = str(response)
            
        self.log_test("OTP Verification", success, details)
        return success and self.citizen_token is not None

    def test_admin_login(self):
        """Test admin login"""
        print("\n🔑 Testing Admin Login...")
        
        login_data = {
            "phone": self.admin_phone,
            "password": self.admin_password
        }
        
        success, response = self.make_request('POST', 'auth/login', data=login_data, expected_status=200)
        
        if success and isinstance(response, dict):
            self.admin_token = response.get('access_token')
            user_info = response.get('user', {})
            details = f"Admin token received, User: {user_info.get('full_name')} ({user_info.get('role')})"
        else:
            details = str(response)
            
        self.log_test("Admin Login", success, details)
        return success and self.admin_token is not None

    def test_citizen_login(self):
        """Test citizen login after registration"""
        print("\n🔑 Testing Citizen Login...")
        
        login_data = {
            "phone": self.test_user_phone,
            "password": "testpass123"
        }
        
        success, response = self.make_request('POST', 'auth/login', data=login_data, expected_status=200)
        
        if success and isinstance(response, dict):
            token = response.get('access_token')
            user_info = response.get('user', {})
            details = f"Citizen login successful, User: {user_info.get('full_name')} ({user_info.get('role')})"
        else:
            details = str(response)
            
        self.log_test("Citizen Login", success, details)
        return success

    def test_create_issue(self):
        """Test issue creation with AI classification"""
        print("\n📝 Testing Issue Creation...")
        
        if not self.citizen_token:
            self.log_test("Issue Creation", False, "No citizen token available")
            return False
        
        # Test with form data (as the API expects)
        issue_data = {
            "title": "Pothole on Main Street causing vehicle damage",
            "description": "Large pothole approximately 2 feet wide and 6 inches deep on Main Street near the intersection with Oak Avenue. Multiple vehicles have been damaged.",
            "category": "pothole",
            "address": "Main Street & Oak Avenue, Downtown",
            "priority": "high"
        }
        
        success, response = self.make_request('POST', 'issues', data=issue_data, token=self.citizen_token, expected_status=200, form_data=True)
        
        if success and isinstance(response, dict):
            issue_id = response.get('id')
            ai_classification = response.get('ai_classification')
            details = f"Issue created with ID: {issue_id}"
            if ai_classification:
                try:
                    ai_data = json.loads(ai_classification)
                    details += f", AI Category: {ai_data.get('category')}, Priority: {ai_data.get('priority')}"
                except:
                    details += f", AI Classification: {ai_classification[:100]}..."
        else:
            details = str(response)
            
        self.log_test("Issue Creation", success, details)
        return success

    def test_get_citizen_issues(self):
        """Test getting issues as citizen"""
        print("\n📋 Testing Get Citizen Issues...")
        
        if not self.citizen_token:
            self.log_test("Get Citizen Issues", False, "No citizen token available")
            return False
        
        success, response = self.make_request('GET', 'issues', token=self.citizen_token, expected_status=200)
        
        if success and isinstance(response, list):
            details = f"Retrieved {len(response)} issues"
            if response:
                first_issue = response[0]
                details += f", First issue: {first_issue.get('title', 'No title')[:50]}..."
        else:
            details = str(response)
            
        self.log_test("Get Citizen Issues", success, details)
        return success

    def test_get_admin_issues(self):
        """Test getting all issues as admin"""
        print("\n📋 Testing Get Admin Issues...")
        
        if not self.admin_token:
            self.log_test("Get Admin Issues", False, "No admin token available")
            return False
        
        success, response = self.make_request('GET', 'issues?limit=100', token=self.admin_token, expected_status=200)
        
        if success and isinstance(response, list):
            details = f"Admin retrieved {len(response)} total issues"
        else:
            details = str(response)
            
        self.log_test("Get Admin Issues", success, details)
        return success

    def test_admin_dashboard(self):
        """Test admin dashboard statistics"""
        print("\n📊 Testing Admin Dashboard...")
        
        if not self.admin_token:
            self.log_test("Admin Dashboard", False, "No admin token available")
            return False
        
        success, response = self.make_request('GET', 'admin/dashboard', token=self.admin_token, expected_status=200)
        
        if success and isinstance(response, dict):
            stats = {
                'total': response.get('total_issues', 0),
                'open': response.get('open_issues', 0),
                'in_progress': response.get('in_progress', 0),
                'resolved': response.get('resolved', 0)
            }
            categories = len(response.get('categories', []))
            priorities = len(response.get('priorities', []))
            details = f"Stats: {stats}, Categories: {categories}, Priorities: {priorities}"
        else:
            details = str(response)
            
        self.log_test("Admin Dashboard", success, details)
        return success

    def test_heatmap_data(self):
        """Test heatmap data endpoint"""
        print("\n🗺️ Testing Heatmap Data...")
        
        if not self.citizen_token:
            self.log_test("Heatmap Data", False, "No citizen token available")
            return False
        
        success, response = self.make_request('GET', 'heatmap-data', token=self.citizen_token, expected_status=200)
        
        if success and isinstance(response, dict):
            data_points = len(response.get('data', []))
            details = f"Retrieved {data_points} heatmap data points"
        else:
            details = str(response)
            
        self.log_test("Heatmap Data", success, details)
        return success

    def test_update_issue_status(self):
        """Test updating issue status (admin only)"""
        print("\n✏️ Testing Issue Status Update...")
        
        if not self.admin_token:
            self.log_test("Issue Status Update", False, "No admin token available")
            return False
        
        # First get an issue to update
        success, issues = self.make_request('GET', 'issues?limit=1', token=self.admin_token, expected_status=200)
        
        if not success or not issues:
            self.log_test("Issue Status Update", False, "No issues available to update")
            return False
        
        issue_id = issues[0]['id']
        update_data = {
            "status": "in_progress",
            "admin_notes": "Issue has been assigned to maintenance team for review"
        }
        
        success, response = self.make_request('PUT', f'issues/{issue_id}', data=update_data, token=self.admin_token, expected_status=200)
        
        if success:
            details = f"Updated issue {issue_id} to in_progress status"
        else:
            details = str(response)
            
        self.log_test("Issue Status Update", success, details)
        return success

    def run_all_tests(self):
        """Run all API tests in sequence"""
        print("🚀 Starting Civic Issue Reporting Platform API Tests")
        print(f"🌐 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Test sequence
        tests = [
            self.test_admin_creation,
            self.test_user_registration,
            self.test_otp_verification,
            self.test_admin_login,
            self.test_citizen_login,
            self.test_create_issue,
            self.test_get_citizen_issues,
            self.test_get_admin_issues,
            self.test_admin_dashboard,
            self.test_heatmap_data,
            self.test_update_issue_status
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(0.5)  # Small delay between tests
            except Exception as e:
                print(f"❌ {test.__name__} - EXCEPTION: {str(e)}")
        
        # Final results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"📈 Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("\n🎉 ALL TESTS PASSED! Backend API is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {self.tests_run - self.tests_passed} tests failed. Please check the issues above.")
            return 1

def main():
    tester = CivicIssueAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())