#!/usr/bin/env python3
"""
Interview Simulation and API Testing Script
Tests the complete Ready4Hire system with a realistic interview scenario
"""

import requests
import json
import time
from typing import Dict, List, Optional
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8001"
API_V2_BASE = f"{API_BASE_URL}/api/v2"

class InterviewSimulator:
    """Simulates a complete interview flow"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.interview_id = None
        self.user_id = f"test_user_{int(time.time())}"
        
    def test_health(self) -> bool:
        """Test if API is healthy"""
        print("🔍 Testing API health...")
        try:
            response = self.session.get(f"{API_V2_BASE}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API is healthy")
                print(f"   Status: {data.get('status')}")
                print(f"   Components:")
                for comp, status in data.get('components', {}).items():
                    print(f"      • {comp}: {status}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Could not connect to API: {str(e)}")
            return False
    
    def start_interview(
        self,
        role: str = "Backend Developer",
        interview_type: str = "technical",
        mode: str = "practice"
    ) -> Optional[str]:
        """Start a new interview"""
        print(f"\n🚀 Starting {interview_type} interview for {role}...")
        
        try:
            payload = {
                "user_id": self.user_id,
                "role": role,
                "type": interview_type,
                "mode": mode
            }
            
            response = self.session.post(
                f"{API_V2_BASE}/interviews",
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                self.interview_id = data.get('interview_id')
                print(f"✅ Interview started: {self.interview_id}")
                
                # Get first question
                first_question = data.get('first_question', {})
                if first_question:
                    print(f"\n📝 First Question:")
                    print(f"   ID: {first_question.get('id', 'N/A')}")
                    print(f"   Q: {first_question.get('text', first_question.get('question', 'N/A'))}")
                    print(f"   Category: {first_question.get('category', 'N/A')}")
                    print(f"   Difficulty: {first_question.get('difficulty', 'N/A')}")
                
                return self.interview_id
            else:
                print(f"❌ Failed to start interview: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error starting interview: {str(e)}")
            return None
    
    def get_next_question(self) -> Optional[Dict]:
        """Get next question in interview"""
        if not self.interview_id:
            print("❌ No active interview")
            return None
        
        print(f"\n🔄 Getting next question...")
        
        try:
            payload = {"user_id": self.user_id}
            response = self.session.post(
                f"{API_V2_BASE}/interviews/{self.interview_id}/next-question",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                question = data.get('question', {})
                print(f"✅ Next question received:")
                print(f"   Q: {question.get('question')}")
                print(f"   Category: {question.get('category')}")
                print(f"   Difficulty: {question.get('difficulty')}")
                return question
            else:
                print(f"⚠️ No more questions or error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting next question: {str(e)}")
            return None
    
    def submit_answer(
        self,
        answer: str,
        question_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Submit answer to current question"""
        if not self.interview_id:
            print("❌ No active interview")
            return None
        
        print(f"\n💬 Submitting answer...")
        print(f"   Answer: {answer[:100]}...")
        
        try:
            payload = {
                "answer": answer,
                "time_taken": 30  # Simulate 30 seconds thinking time
            }
            
            response = self.session.post(
                f"{API_V2_BASE}/interviews/{self.interview_id}/answers",
                json=payload,
                timeout=120  # Evaluation can take time with LLM
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract evaluation info
                score = data.get('score', 'N/A')
                feedback = data.get('feedback', 'N/A')
                
                print(f"✅ Answer evaluated:")
                print(f"   Score: {score}/10")
                print(f"   Feedback: {feedback[:150]}...")
                
                next_q = data.get('next_question')
                if next_q:
                    print(f"\n📝 Next Question:")
                    print(f"   Q: {next_q.get('text', next_q.get('question', 'N/A'))[:100]}...")
                else:
                    print(f"\n✅ Interview complete or no more questions")
                
                return data
            else:
                print(f"❌ Failed to submit answer: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error submitting answer: {str(e)}")
            return None
    
    def end_interview(self) -> Optional[Dict]:
        """End interview and get final evaluation"""
        if not self.interview_id:
            print("❌ No active interview")
            return None
        
        print(f"\n🏁 Ending interview...")
        
        try:
            response = self.session.post(
                f"{API_V2_BASE}/interviews/{self.interview_id}/end",
                json={"user_id": self.user_id},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get('summary', {})
                
                print(f"✅ Interview completed!")
                print(f"\n📊 Final Results:")
                print(f"   Overall Score: {summary.get('overall_score', 'N/A')}/10")
                print(f"   Questions Answered: {summary.get('questions_answered', 0)}")
                print(f"   Average Score: {summary.get('average_score', 'N/A')}")
                print(f"   Duration: {summary.get('duration', 'N/A')}")
                
                strengths = summary.get('strengths', [])
                if strengths:
                    print(f"\n💪 Strengths:")
                    for strength in strengths[:3]:
                        print(f"      • {strength}")
                
                improvements = summary.get('areas_for_improvement', [])
                if improvements:
                    print(f"\n📈 Areas for Improvement:")
                    for area in improvements[:3]:
                        print(f"      • {area}")
                
                return data
            else:
                print(f"❌ Failed to end interview: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error ending interview: {str(e)}")
            return None
    
    def get_interview_history(self) -> Optional[Dict]:
        """Get interview history"""
        print(f"\n📜 Getting interview history...")
        
        try:
            response = self.session.get(
                f"{API_V2_BASE}/interviews/history",
                params={"user_id": self.user_id},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                interviews = data.get('interviews', [])
                print(f"✅ Found {len(interviews)} interview(s)")
                return data
            else:
                print(f"⚠️ Could not get history: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting history: {str(e)}")
            return None


def run_complete_simulation():
    """Run a complete interview simulation"""
    print("=" * 70)
    print("  Ready4Hire - Complete Interview Simulation")
    print("=" * 70)
    
    simulator = InterviewSimulator()
    
    # Test health
    if not simulator.test_health():
        print("\n❌ API is not available. Please start the services first.")
        print("   Run: ./scripts/run.sh --dev")
        return
    
    # Start interview
    interview_id = simulator.start_interview(
        role="Backend Developer",
        interview_type="technical",
        mode="practice"
    )
    
    if not interview_id:
        print("\n❌ Could not start interview")
        return
    
    # Sample answers for testing
    sample_answers = [
        "SOLID principles are five design principles in object-oriented programming: Single Responsibility Principle (each class should have one responsibility), Open-Closed Principle (open for extension, closed for modification), Liskov Substitution Principle (subtypes should be substitutable for their base types), Interface Segregation Principle (clients shouldn't depend on interfaces they don't use), and Dependency Inversion Principle (depend on abstractions, not concretions).",
        
        "A RESTful API is an architectural style for web services that uses HTTP methods. It's stateless, cacheable, and uses standard HTTP status codes. Resources are identified by URLs, and operations are performed using GET, POST, PUT, DELETE methods. It's different from SOAP which is protocol-based and more complex.",
        
        "To optimize database queries, I would: 1) Add appropriate indexes on frequently queried columns, 2) Use EXPLAIN to analyze query execution plans, 3) Avoid N+1 queries by using joins or eager loading, 4) Implement caching for frequently accessed data, 5) Use pagination for large result sets, 6) Optimize the database schema and normalize data appropriately.",
    ]
    
    # Simulate answering questions
    for i, answer in enumerate(sample_answers, 1):
        time.sleep(1)  # Simulate thinking time
        result = simulator.submit_answer(answer)
        
        if not result:
            print(f"\n⚠️ Could not submit answer {i}")
            break
        
        # Check if interview ended
        if result.get('status') == 'completed':
            break
    
    # End interview
    time.sleep(1)
    final_results = simulator.end_interview()
    
    # Get history
    time.sleep(1)
    simulator.get_interview_history()
    
    print("\n" + "=" * 70)
    print("  Simulation Complete!")
    print("=" * 70)


def test_additional_endpoints():
    """Test additional API endpoints"""
    print("\n" + "=" * 70)
    print("  Testing Additional Endpoints")
    print("=" * 70)
    
    session = requests.Session()
    
    # Test metrics endpoint
    print("\n🔍 Testing GET /api/v2/metrics...")
    try:
        response = session.get(f"{API_V2_BASE}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print(f"✅ Metrics retrieved:")
            print(f"   Total Interviews: {metrics.get('total_interviews', 'N/A')}")
            print(f"   Avg Score: {metrics.get('average_score', 'N/A')}")
            print(f"   Questions Asked: {metrics.get('questions_asked', 'N/A')}")
        else:
            print(f"⚠️ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test root endpoint
    print("\n🔍 Testing GET / (root)...")
    try:
        response = session.get(API_BASE_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint responding:")
            print(f"   {data}")
        else:
            print(f"⚠️ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    print(f"\n{'='*70}")
    print(f"  Ready4Hire API Testing Suite")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    # Run main simulation
    run_complete_simulation()
    
    # Test additional endpoints
    test_additional_endpoints()
    
    print(f"\n{'='*70}")
    print(f"  All Tests Complete!")
    print(f"{'='*70}\n")
