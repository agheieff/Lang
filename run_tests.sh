#!/bin/bash

# Arcadia Lang Test Runner
# A comprehensive testing script for the Arcadia Lang application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
UNIT_TESTS=true
INTEGRATION_TESTS=true
COVERAGE=false
VERBOSE=false
WATCH=false
SPECIFIC_TEST=""

# Help function
show_help() {
    echo "Arcadia Lang Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -u, --unit-only        Run only unit tests"
    echo "  -i, --integration-only  Run only integration tests"  
    echo "  -c, --coverage         Generate coverage report"
    echo "  -v, --verbose          Verbose output"
    echo "  -w, --watch            Watch mode (run tests on file changes)"
    echo "  -t, --test TEST        Run specific test file or function"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     Run all tests"
    echo "  $0 -u                  Run only unit tests"
    echo "  $0 -c                  Run all tests with coverage"
    echo "  $0 -t test_srs         Run only SRS-related tests"
    echo "  $0 -t tests/unit/test_srs_service.py::test_add_lexeme"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit-only)
            UNIT_TESTS=true
            INTEGRATION_TESTS=false
            shift
            ;;
        -i|--integration-only)
            UNIT_TESTS=false
            INTEGRATION_TESTS=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -w|--watch)
            WATCH=true
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Base pytest command
PYTEST_CMD="uv run pytest"

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage if requested
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=server --cov-report=html --cov-report=term-missing"
fi

# Add watch mode if requested
if [ "$WATCH" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --watch"
fi

# Function to run tests with proper output
run_tests() {
    local test_type=$1
    local test_path=$2
    local marker=$3
    
    echo -e "${BLUE}Running $test_type tests...${NC}"
    
    if [ -n "$SPECIFIC_TEST" ]; then
        echo -e "${YELLOW}Running specific test: $SPECIFIC_TEST${NC}"
        if [[ "$SPECIFIC_TEST" == "auth" ]]; then
            $PYTEST_CMD "tests/unit/auth/"
        else
            $PYTEST_CMD "$SPECIFIC_TEST"
        fi
    elif [ -n "$marker" ]; then
        $PYTEST_CMD "$test_path"
    else
        $PYTEST_CMD "$test_path"
    fi
    
    local result=$?
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✅ $test_type tests passed!${NC}"
    else
        echo -e "${RED}❌ $test_type tests failed!${NC}"
        exit 1
    fi
}

# Main execution
echo -e "${BLUE}🚀 Arcadia Lang Test Runner${NC}"
echo -e "${BLUE}================================${NC}"

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must be run from project root (where pyproject.toml is located)${NC}"
    exit 1
fi

# Install test dependencies if needed
echo -e "${YELLOW}Checking test dependencies...${NC}"
uv add --dev pytest pytest-asyncio pytest-cov pytest-mock pytest-watch > /dev/null 2>&1 || true

# Create test database if needed
echo -e "${YELLOW}Setting up test environment...${NC}"
export ARC_LANG_ENVIRONMENT=test
export ARC_LANG_JWT_SECRET=test-secret

# Run tests based on options
if [ "$UNIT_TESTS" = true ]; then
    if [ -n "$SPECIFIC_TEST" ]; then
        # Let the specific test logic handle it
        echo -e "${BLUE}Running specific test...${NC}"
        # This is handled in the run_tests function when SPECIFIC_TEST is set
    else
        run_tests "Unit" "tests/unit" "unit"
    fi
fi

if [ "$INTEGRATION_TESTS" = true ]; then
    run_tests "Integration" "tests/integration" "integration"
fi

# Show coverage report if generated
if [ "$COVERAGE" = true ] && [ -d "htmlcov" ]; then
    echo -e "${BLUE}Coverage report generated in htmlcov/index.html${NC}"
fi

echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
