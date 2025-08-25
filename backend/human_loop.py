"""
Human-in-the-loop features for ESG investment decisions.

This module implements features that allow human oversight and
intervention in AI-generated investment recommendations.
"""

import json
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from dataclasses import dataclass, asdict


@dataclass
class ApprovalRequest:
    """Data structure for approval requests."""
    
    request_id: str
    user_id: str
    recommendation_type: str  # 'portfolio', 'esg_score', 'risk_assessment'
    recommendation_data: Dict[str, Any]
    requested_by: str
    requested_at: str
    status: str = "pending"  # pending, approved, rejected, expired
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    priority: str = "normal"  # low, normal, high, urgent


@dataclass
class HumanDecision:
    """Data structure for human decisions."""
    
    decision_id: str
    request_id: str
    decision: str  # 'approve', 'reject', 'modify'
    decision_maker: str
    decision_at: str
    reasoning: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None


class HumanLoop:
    """
    Human-in-the-loop workflow management using Portia AI.
    
    This class manages approval workflows, human feedback integration,
    and audit trails for AI-generated investment recommendations.
    """
    
    def __init__(self, portia_api_key: str, base_url: str = "https://api.portialabs.ai"):
        """
        Initialize HumanLoop with Portia API credentials.
        
        Args:
            portia_api_key: Portia AI API key
            base_url: Portia API base URL
        """
        self.portia_api_key = portia_api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {portia_api_key}",
            "Content-Type": "application/json"
        }
        
        # In-memory storage for approval requests and decisions
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.human_decisions: Dict[str, HumanDecision] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Load configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from settings.json file."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: Configuration file not found, using defaults")
            return {}
        except json.JSONDecodeError:
            print("Warning: Invalid configuration file, using defaults")
            return {}
    
    def create_approval_request(self, user_id: str, recommendation_type: str, 
                               recommendation_data: Dict[str, Any], 
                               requested_by: str, priority: str = "normal") -> str:
        """
        Create a new approval request.
        
        Args:
            user_id: ID of the user requesting approval
            recommendation_type: Type of recommendation
            recommendation_data: Data to be approved
            requested_by: ID of the person requesting approval
            priority: Priority level of the request
            
        Returns:
            Request ID for tracking
        """
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        approval_request = ApprovalRequest(
            request_id=request_id,
            user_id=user_id,
            recommendation_type=recommendation_type,
            recommendation_data=recommendation_data,
            requested_by=requested_by,
            requested_at=datetime.now().isoformat(),
            priority=priority
        )
        
        self.approval_requests[request_id] = approval_request
        
        # Add to audit trail
        self._add_audit_entry("approval_request_created", {
            "request_id": request_id,
            "user_id": user_id,
            "type": recommendation_type,
            "priority": priority
        })
        
        return request_id
    
    def request_approval(self, payload: Dict[str, Any]) -> bool:
        """
        Request approval using Portia AI.
        
        Args:
            payload: Data payload for approval request
            
        Returns:
            True if approval workflow was initiated successfully, False otherwise
        """
        try:
            # Extract required fields from payload
            user_id = payload.get("user_id")
            recommendation_type = payload.get("recommendation_type")
            recommendation_data = payload.get("recommendation_data")
            requested_by = payload.get("requested_by", "system")
            priority = payload.get("priority", "normal")
            
            if not all([user_id, recommendation_type, recommendation_data]):
                raise ValueError("Missing required fields in payload")
            
            # Create approval request
            request_id = self.create_approval_request(
                user_id=user_id,
                recommendation_type=recommendation_type,
                recommendation_data=recommendation_data,
                requested_by=requested_by,
                priority=priority
            )
            
            # Create Portia plan for approval workflow
            portia_plan = self._create_portia_approval_plan(request_id, payload)
            
            # Send to Portia for processing
            success = self._send_to_portia(portia_plan)
            
            if success:
                # Poll for decision
                decision = self._poll_for_decision(request_id)
                return decision
            else:
                return False
                
        except Exception as e:
            self._add_audit_entry("approval_request_failed", {
                "error": str(e),
                "payload": payload
            })
            return False
    
    def _create_portia_approval_plan(self, request_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Portia plan for approval workflow.
        
        Args:
            request_id: Unique request ID
            payload: Request payload data
            
        Returns:
            Portia plan structure
        """
        # Create a simple approval plan with conditional logic
        plan = {
            "plan_id": f"approval_{request_id}",
            "description": f"Human approval workflow for {payload.get('recommendation_type', 'investment')} recommendation",
            "steps": [
                {
                    "step_id": "review_recommendation",
                    "type": "human_review",
                    "description": "Review AI-generated investment recommendation",
                    "data": payload,
                    "required_fields": ["user_id", "recommendation_type", "recommendation_data"],
                    "approval_threshold": 1,  # Requires 1 human approval
                    "timeout_hours": 24
                },
                {
                    "step_id": "make_decision",
                    "type": "conditional",
                    "description": "Make approval decision based on human review",
                    "conditions": [
                        {
                            "if": "review_recommendation.approved == True",
                            "then": "approve_recommendation",
                            "else": "reject_recommendation"
                        }
                    ]
                },
                {
                    "step_id": "approve_recommendation",
                    "type": "action",
                    "description": "Approve and execute recommendation",
                    "action": "approve",
                    "next_step": "complete_workflow"
                },
                {
                    "step_id": "reject_recommendation",
                    "type": "action",
                    "description": "Reject recommendation with feedback",
                    "action": "reject",
                    "next_step": "complete_workflow"
                },
                {
                    "step_id": "complete_workflow",
                    "type": "completion",
                    "description": "Complete approval workflow",
                    "output": "workflow_completed"
                }
            ],
            "metadata": {
                "request_id": request_id,
                "created_at": datetime.now().isoformat(),
                "priority": payload.get("priority", "normal"),
                "workflow_type": "human_approval"
            }
        }
        
        return plan
    
    def _send_to_portia(self, plan: Dict[str, Any]) -> bool:
        """
        Send approval plan to Portia AI.
        
        Args:
            plan: Portia plan structure
            
        Returns:
            True if successfully sent, False otherwise
        """
        try:
            # In a real implementation, you would send this to Portia's API
            # For now, we'll simulate the API call
            
            # Simulate API endpoint
            endpoint = f"{self.base_url}/v1/plans"
            
            # Simulate API call (replace with actual Portia API call)
            print(f"Simulating Portia API call to: {endpoint}")
            print(f"Plan: {json.dumps(plan, indent=2)}")
            
            # For demo purposes, assume success
            # In production, you would make an actual HTTP request:
            # response = requests.post(endpoint, headers=self.headers, json=plan)
            # return response.status_code == 200
            
            return True
            
        except Exception as e:
            self._add_audit_entry("portia_api_call_failed", {
                "error": str(e),
                "plan_id": plan.get("plan_id")
            })
            return False
    
    def _poll_for_decision(self, request_id: str, max_wait_time: int = 300) -> bool:
        """
        Poll for decision from Portia workflow.
        
        Args:
            request_id: Request ID to poll for
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if approved, False if rejected or timeout
        """
        start_time = time.time()
        poll_interval = 5  # Poll every 5 seconds
        
        while time.time() - start_time < max_wait_time:
            # Check if we have a decision
            if request_id in self.human_decisions:
                decision = self.human_decisions[request_id]
                return decision.decision == "approve"
            
            # Simulate waiting for human decision
            time.sleep(poll_interval)
        
        # Timeout - mark as expired
        if request_id in self.approval_requests:
            self.approval_requests[request_id].status = "expired"
        
        self._add_audit_entry("approval_timeout", {
            "request_id": request_id,
            "max_wait_time": max_wait_time
        })
        
        return False
    
    def record_human_decision(self, request_id: str, decision: str, 
                             decision_maker: str, reasoning: Optional[str] = None,
                             modifications: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a human decision for an approval request.
        
        Args:
            request_id: ID of the approval request
            decision: Decision made ('approve', 'reject', 'modify')
            decision_maker: ID of the person making the decision
            reasoning: Optional reasoning for the decision
            modifications: Optional modifications to the recommendation
            
        Returns:
            Decision ID
        """
        if request_id not in self.approval_requests:
            raise ValueError(f"Approval request {request_id} not found")
        
        decision_id = f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}"
        
        human_decision = HumanDecision(
            decision_id=decision_id,
            request_id=request_id,
            decision=decision,
            decision_maker=decision_maker,
            decision_at=datetime.now().isoformat(),
            reasoning=reasoning,
            modifications=modifications
        )
        
        self.human_decisions[decision_id] = human_decision
        
        # Update approval request status
        approval_request = self.approval_requests[request_id]
        if decision == "approve":
            approval_request.status = "approved"
            approval_request.approved_by = decision_maker
            approval_request.approved_at = datetime.now().isoformat()
        elif decision == "reject":
            approval_request.status = "rejected"
            approval_request.rejection_reason = reasoning
        
        # Add to audit trail
        self._add_audit_entry("human_decision_recorded", {
            "request_id": request_id,
            "decision_id": decision_id,
            "decision": decision,
            "decision_maker": decision_maker,
            "reasoning": reasoning
        })
        
        return decision_id
    
    def get_approval_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an approval request.
        
        Args:
            request_id: ID of the approval request
            
        Returns:
            Dictionary with approval status information
        """
        if request_id not in self.approval_requests:
            return None
        
        approval_request = self.approval_requests[request_id]
        decision = None
        
        # Find associated decision if any
        for dec_id, dec in self.human_decisions.items():
            if dec.request_id == request_id:
                decision = asdict(dec)
                break
        
        return {
            "request": asdict(approval_request),
            "decision": decision,
            "status": approval_request.status
        }
    
    def get_pending_approvals(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of pending approval requests.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of pending approval requests
        """
        pending = []
        
        for request_id, request in self.approval_requests.items():
            if request.status == "pending":
                if user_id is None or request.user_id == user_id:
                    pending.append(asdict(request))
        
        return pending
    
    def _add_audit_entry(self, action: str, data: Dict[str, Any]) -> None:
        """
        Add entry to audit trail.
        
        Args:
            action: Action performed
            data: Associated data
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data
        }
        
        self.audit_trail.append(audit_entry)
    
    def get_audit_trail(self, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get audit trail entries.
        
        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            
        Returns:
            List of audit trail entries
        """
        if not start_date and not end_date:
            return self.audit_trail.copy()
        
        filtered_entries = []
        
        for entry in self.audit_trail:
            entry_date = entry["timestamp"]
            
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue
            
            filtered_entries.append(entry)
        
        return filtered_entries
    
    def export_audit_data(self) -> Dict[str, Any]:
        """
        Export all audit data for backup or analysis.
        
        Returns:
            Dictionary containing all audit data
        """
        return {
            "approval_requests": {k: asdict(v) for k, v in self.approval_requests.items()},
            "human_decisions": {k: asdict(v) for k, v in self.human_decisions.items()},
            "audit_trail": self.audit_trail,
            "exported_at": datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize HumanLoop (you would use actual API key)
    human_loop = HumanLoop(portia_api_key="demo_key")
    
    # Example approval request
    payload = {
        "user_id": "user123",
        "recommendation_type": "portfolio",
        "recommendation_data": {
            "stocks": ["AAPL", "MSFT", "GOOGL"],
            "allocation": [0.4, 0.3, 0.3],
            "esg_score": 85.5
        },
        "requested_by": "ai_system",
        "priority": "normal"
    }
    
    print("Creating approval request...")
    success = human_loop.request_approval(payload)
    print(f"Approval workflow initiated: {success}")
    
    # Simulate human decision
    print("\nSimulating human decision...")
    request_id = list(human_loop.approval_requests.keys())[0]
    decision_id = human_loop.record_human_decision(
        request_id=request_id,
        decision="approve",
        decision_maker="portfolio_manager",
        reasoning="Recommendation meets ESG criteria and risk tolerance"
    )
    
    print(f"Human decision recorded: {decision_id}")
    
    # Check approval status
    status = human_loop.get_approval_status(request_id)
    print(f"\nApproval status: {status['status']}")
    
    # Get audit trail
    audit = human_loop.get_audit_trail()
    print(f"\nAudit trail entries: {len(audit)}")
    
    # Export data
    export_data = human_loop.export_audit_data()
    print(f"\nExported data keys: {list(export_data.keys())}")
