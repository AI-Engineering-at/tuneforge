# TODO: Unit Test Coverage - finetune/zeroth_core.py

**Target:** 100% coverage (currently ~45%)  
**Priority:** P0  
**Estimate:** 1 day

## Test Cases

### Dataset Hashing (`_hash_dataset`)
- [ ] `test_hash_empty_dataset` - Empty list returns consistent hash
- [ ] `test_hash_deterministic` - Same input → same hash
- [ ] `test_hash_order_sensitive` - Different order → different hash
- [ ] `test_hash_content_sensitive` - Modified content → different hash
- [ ] `test_hash_large_dataset` - Performance with 10k+ records

### Tag Extraction (`_extract_tags`)
- [ ] `test_extract_tags_from_valid_config` - Normal config with tags
- [ ] `test_extract_tags_missing_field` - Config without tags field
- [ ] `test_extract_tags_empty_list` - Config with empty tags
- [ ] `test_extract_tags_non_list` - Config with string instead of list

### Pre-Train Check (`pre_train_zeroth_check`)
- [ ] `test_pre_train_success` - Valid config/dataset returns clearance
- [ ] `test_pre_train_with_state_machine` - State transitions on success
- [ ] `test_pre_train_clears_cache` - Cache cleared before check

### Error Handling
- [ ] `test_connection_error_transitions_state` - DEGRADED_AEGIS state
- [ ] `test_connection_error_raises_exception` - ZerothConnectionError raised
- [ ] `test_permission_error_transitions_state` - HALTED_AEGIS_DENY state
- [ ] `test_permission_error_raises_exception` - ZerothAccessDeniedError raised
- [ ] `test_connection_error_logs_error` - Structured logging on failure
- [ ] `test_permission_error_logs_error` - Structured logging on denial

### Pre-Publish Check (`pre_publish_zeroth_check`)
- [ ] `test_pre_publish_success` - Valid model card/manifest returns clearance
- [ ] `test_pre_publish_with_attestation` - Includes tester attestation
- [ ] `test_pre_publish_missing_validation_report` - Warning if no report

### Exception Classes
- [ ] `test_access_denied_error_message` - Error message formatting
- [ ] `test_connection_error_message` - Error message formatting
- [ ] `test_exceptions_inheritable` - Can catch both with ZerothError

## Mock Strategy

```python
# Mock AegisClient for unit tests
@pytest.fixture
def mock_aegis_success():
    with patch('finetune.zeroth_core._client') as mock:
        mock.evaluate_policy.return_value = {
            "clearance_token": "test-token-123",
            "status": "approved"
        }
        yield mock

@pytest.fixture
def mock_aegis_connection_error():
    with patch('finetune.zeroth_core._client') as mock:
        mock.evaluate_policy.side_effect = ConnectionError("Aegis unreachable")
        yield mock

@pytest.fixture
def mock_aegis_permission_error():
    with patch('finetune.zeroth_core._client') as mock:
        mock.evaluate_policy.side_effect = PermissionError("Access denied")
        yield mock
```

## Acceptance Criteria
- [ ] 100% line coverage
- [ ] All branches covered
- [ ] Edge cases documented
- [ ] Tests run in <1s
