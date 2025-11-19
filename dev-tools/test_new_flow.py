#!/usr/bin/env python3
"""
Simple test runner for the new generation flow.
This checks that all the pieces can be loaded and wired together correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test that all new modules can be imported."""
    from server.services.state_manager import GenerationStateManager, TextState  # noqa: F401
    from server.services.text_generation_service import TextGenerationService, TextGenerationResult  # noqa: F401
    from server.services.notification_service import NotificationService, get_notification_service  # noqa: F401
    from server.services.generation_orchestrator import GenerationOrchestrator  # noqa: F401
    print("✅ All new services imported successfully")

def test_state_manager():
    """Test basic state manager functionality."""
    from server.services.state_manager import TextState

    # Test enum values
    assert TextState.GENERATING == "generating"
    assert TextState.CONTENT_READY == "content_ready"
    assert TextState.FULLY_READY == "fully_ready"
    assert TextState.FAILED == "failed"

    print("✅ State manager enums work correctly")

def test_notification_service():
    """Test notification service singleton pattern."""
    from server.services.notification_service import get_notification_service

    # Check that we get the same instance
    service1 = get_notification_service()
    service2 = get_notification_service()
    assert service1 is service2

    print("✅ Notification service singleton works correctly")

def test_orchestrator():
    """Test that orchestrator wires services together."""
    from server.services.generation_orchestrator import GenerationOrchestrator

    orchestrator = GenerationOrchestrator()

    # Should have all required services
    assert hasattr(orchestrator, 'state_manager')
    assert hasattr(orchestrator, 'text_gen_service')
    assert hasattr(orchestrator, 'notification_service')
    assert hasattr(orchestrator, 'retry_service')
    assert hasattr(orchestrator, 'readiness_service')

    print("✅ Generation orchestrator has all required services")

def test_routes():
    """Test that reading routes have the new endpoints."""
    from server.routes.reading import router

    routes = [route.path for route in router.routes]

    # Check for new SSE endpoint
    assert '/reading/events/sse' in routes, "Missing SSE endpoint"

    # Check that original endpoints are still there
    assert '/reading/current' in routes
    assert '/reading/next' in routes
    assert '/reading/{text_id}/words' in routes

    print("✅ Reading routes have all required endpoints")

def test_app():
    """Test that the main FastAPI app can be created."""
    from server.main import app
    from fastapi.routing import APIRoute

    # Get all route paths
    routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.append(route.path)

    # Check for our SSE endpoint
    assert '/reading/events/sse' in routes, "SSE endpoint not registered with main app"

    print("✅ Main app has all routes registered")

def test_static_files():
    """Test that static files exist."""
    static_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'server', 'static'))

    # Check for new SSE script
    sse_script = os.path.join(static_dir, 'reading-sse.js')
    assert os.path.exists(sse_script), f"SSE script not found: {sse_script}"

    # Check it's not empty
    content = open(sse_script, encoding="utf-8").read()
    assert len(content) >= 1000, f"SSE script seems too small ({len(content)} bytes)"

    # Check for existing reading script
    reading_script = os.path.join(static_dir, 'reading.js')
    assert os.path.exists(reading_script), f"Reading script not found: {reading_script}"

    print("✅ Static files are present")

def main():
    """Run all tests."""
    tests = [
        test_imports,
        test_state_manager,
        test_notification_service,
        test_orchestrator,
        test_routes,
        test_app,
        test_static_files
    ]
    
    passed = 0
    failed = 0
    
    print("Running new generation flow tests...")
    print()
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! The new generation flow is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
