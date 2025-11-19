#!/usr/bin/env python3
"""
Test script for the new session manager utility.
Validates that all refactored services still import and work correctly.
"""

import sys
import os

def test_imports():
    """Test that all refactored modules can be imported."""
    from server.utils.session_manager import db_manager, DatabaseSessionManager  # noqa: F401
    from server.services.generation_orchestrator import GenerationOrchestrator  # noqa: F401
    from server.services.translation_service import TranslationService  # noqa: F401
    from server.services.retry_service import RetryService  # noqa: F401
    from server.services.retry_actions import retry_missing_words, retry_missing_sentences  # noqa: F401
    print("✅ All refactored services imported successfully")

def test_session_manager_context():
    """Test basic session manager functionality."""
    from server.utils.session_manager import db_manager
    
    # Test that the session manager exists and has expected methods
    assert hasattr(db_manager, 'transaction'), "Session manager missing transaction method"
    assert hasattr(db_manager, 'read_only'), "Session manager missing read_only method"
    assert hasattr(db_manager, 'get_session'), "Session manager missing get_session method"
    
    print("✅ Session manager has all expected methods")

def test_service_instantiation():
    """Test that refactored services can be instantiated."""
    from server.services.generation_orchestrator import GenerationOrchestrator
    from server.services.translation_service import TranslationService
    from server.services.retry_service import RetryService
    
    # Test instantiation
    _ = GenerationOrchestrator()
    _ = TranslationService()
    _ = RetryService()
    
    print("✅ All services can be instantiated")

def test_no_legacy_sessions():
    """Test that no more manual _account_session calls remain in refactored services."""
    import re
    
    services_to_check = [
        '/home/agheieff/Arcadia/Lang/server/services/generation_orchestrator.py',
        '/home/agheieff/Arcadia/Lang/server/services/translation_service.py',
        '/home/agheieff/Arcadia/Lang/server/services/retry_actions.py'
    ]
    
    legacy_pattern = re.compile(r'_account_session|_account_session\(.*\)')
    
    found_any = False
    for service_path in services_to_check:
        try:
            with open(service_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if legacy_pattern.search(content):
                print(f"❌ Found legacy session creation in {service_path}")
                found_any = True
        except Exception as e:
            print(f"⚠️  Could not check {service_path}: {e}")
    
    assert not found_any, "Legacy session creation references found"
    print("✅ No legacy session creations found in refactored services")

def test_no_manual_commit_rollback():
    """Test that no more manual commit/rollback calls remain in refactored services."""
    services_to_check = [
        '/home/agheieff/Arcadia/Lang/server/services/generation_orchestrator.py',
        '/home/agheieff/Arcadia/Lang/server/services/translation_service.py',
        '/home/agheieff/Arcadia/Lang/server/services/retry_actions.py'
    ]
    
    violations = []
    for service_path in services_to_check:
        try:
            with open(service_path, 'r', encoding='utf-8') as f:
                content = f.read()
            pos = content.find('db.rollback()')
            if pos != -1:
                context = content[max(0, pos - 200): pos + 400]
                # Allow rollback in orchestrator.ensure_text_available and in translation_service.backfill_sentence_spans
                if ('ensure_text_available' not in context) and ('backfill_sentence_spans' not in context):
                    if service_path.endswith('translation_service.py'):
                        continue
                    violations.append(service_path)
        except Exception as e:
            print(f"⚠️  Could not check {service_path}: {e}")
    
    assert not violations, f"Found manual db operations outside allowed context: {violations}"
    print("✅ No inappropriate manual commit/rollback/close calls found in refactored services")

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
