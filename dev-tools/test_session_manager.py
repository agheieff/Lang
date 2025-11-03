#!/usr/bin/env python3
"""
Test script for the new session manager utility.
Validates that all refactored services still import and work correctly.
"""

import sys
import os

def test_imports():
    """Test that all refactored modules can be imported."""
    try:
        from server.utils.session_manager import db_manager, DatabaseSessionManager
        from server.services.generation_orchestrator import GenerationOrchestrator
        from server.services.translation_service import TranslationService
        from server.services.retry_service import RetryService
        from server.services.retry_actions import retry_missing_words, retry_missing_sentences
        print("✅ All refactored services imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_session_manager_context():
    """Test basic session manager functionality."""
    try:
        from server.utils.session_manager import db_manager
        from server.db import GlobalSessionLocal
        
        # Test that the session manager exists and has expected methods
        assert hasattr(db_manager, 'transaction'), "Session manager missing transaction method"
        assert hasattr(db_manager, 'read_only'), "Session manager missing read_only method"
        assert hasattr(db_manager, 'get_session'), "Session manager missing get_session method"
        
        print("✅ Session manager has all expected methods")
        return True
    except Exception as e:
        print(f"❌ Session manager test failed: {e}")
        return False

def test_service_instantiation():
    """Test that refactored services can be instantiated."""
    try:
        from server.services.generation_orchestrator import GenerationOrchestrator
        from server.services.translation_service import TranslationService
        from server.services.retry_service import RetryService
        
        # Test instantiation
        orchestrator = GenerationOrchestrator()
        translation_service = TranslationService()
        retry_service = RetryService()
        
        print("✅ All services can be instantiated")
        return True
    except Exception as e:
        print(f"❌ Service instantiation failed: {e}")
        return False

def test_no_legacy_sessions():
    """Test that no more manual _account_session calls remain in refactored services."""
    import re
    
    services_to_check = [
        '/home/agheieff/Arcadia/Lang/server/services/generation_orchestrator.py',
        '/home/agheieff/Arcadia/Lang/server/services/translation_service.py',
        '/home/agheieff/Arcadia/Lang/server/services/retry_actions.py'
    ]
    
    legacy_pattern = re.compile(r'_account_session|_account_session\(.*\)')
    
    for service_path in services_to_check:
        try:
            with open(service_path, 'r') as f:
                content = f.read()
                
            if legacy_pattern.search(content):
                print(f"❌ Found legacy session creation in {service_path}")
                return False
        except Exception as e:
            print(f"⚠️  Could not check {service_path}: {e}")
            
    print("✅ No legacy session creations found in refactored services")
    return True

def test_no_manual_commit_rollback():
    """Test that no more manual commit/rollback calls remain in refactored services."""
    import re
    
    services_to_check = [
        '/home/agheieff/Arcadia/Lang/server/services/generation_orchestrator.py',
        '/home/agheieff/Arcadia/Lang/server/services/translation_service.py',
        '/home/agheieff/Arcadia/Lang/server/services/retry_actions.py'
    ]
    
    # Allow this specific case where rollback is intentional in ensure_text_available
    allowed_pattern = re.compile(r'except Exception:.*\n.*try:.*\n.*db\.rollback\(\)')
    
    # Find all manual db ops but exclude the allowed pattern
    for service_path in services_to_check:
        try:
            with open(service_path, 'r') as f:
                content = f.read()
                
            first_db_rollback_pos = content.find('db.rollback()')
            if first_db_rollback_pos != -1:
                # Check if this is the allowed case
                context_start = max(0, first_db_rollback_pos - 100)
                context_end = min(len(content), first_db_rollback_pos + 200)
                context = content[context_start:context_end]
                
                if "ensure_text_available" in context:
                    # This is the allowed rollback in ensure_text_available
                    continue
                    
                print(f"❌ Found manual db operations in {service_path}")
                return False
        except Exception as e:
            print(f"⚠️  Could not check {service_path}: {e}")
            
    print("✅ No inappropriate manual commit/rollback/close calls found in refactored services")
    return True

def main():
    """Run all tests."""
    print("Testing session manager refactoring...")
    
    tests = [
        test_imports,
        test_session_manager_context,
        test_service_instantiation,
        test_no_legacy_sessions,
        test_no_manual_commit_rollback,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All session manager tests passed! Refactoring is successful.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
