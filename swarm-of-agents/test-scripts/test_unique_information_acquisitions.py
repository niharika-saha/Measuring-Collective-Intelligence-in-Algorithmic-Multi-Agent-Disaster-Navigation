from types import SimpleNamespace

from collective_metrics import useful_information_metrics


def sim(shared=(), messages=(), discoveries=(), event_map=None):
    return SimpleNamespace(
        _shared_memory_reads=list(shared),
        _local_message_reads=list(messages),
        _hazard_discoveries=list(discoveries),
        _event_ids_by_node=event_map or {},
    )


def test_duplicate_messages_and_repeated_board_reads_count_once_as_acquisition():
    messages = [
        {"reader_agent": 1, "node_id": "n1", "cross_agent": True,
         "channel": "teammate_message", "retrieval_step": 2,
         "route_relevant": True, "new_information": True},
        {"reader_agent": 1, "node_id": "n1", "cross_agent": True,
         "channel": "teammate_message", "retrieval_step": 3,
         "route_relevant": False, "new_information": False},
    ]
    shared = [
        {"reader_agent": 1, "node_id": "n1", "cross_agent": True,
         "channel": "blocked_nodes", "retrieval_step": 4,
         "route_relevant": False},
    ]
    result = useful_information_metrics(sim(shared, messages))
    assert result["cross_agent_hazard_information_reads"] == 3
    assert result["unique_recipient_node_acquisitions"] == 1
    assert result["unique_route_relevant_recipient_node_acquisitions"] == 1
    assert result["repeated_hazard_reads_after_first_recipient_node_acquisition"] == 2
    assert result["unique_acquisition_route_relevance_ratio"] == 1.0


def test_first_acquisition_is_deduplicated_across_channels_and_snapshots_relevance():
    messages = [{
        "reader_agent": 2, "node_id": "n2", "cross_agent": True,
        "channel": "teammate_message", "retrieval_step": 5,
        "route_relevant": False, "new_information": True,
    }]
    shared = [{
        "reader_agent": 2, "node_id": "n2", "cross_agent": True,
        "channel": "danger_nodes", "retrieval_step": 6,
        "route_relevant": True,
    }]
    result = useful_information_metrics(sim(shared, messages))
    assert result["unique_recipient_node_acquisitions"] == 1
    assert result["unique_route_relevant_recipient_node_acquisitions"] == 0


def test_event_level_acquisitions_deduplicate_multiple_nodes_for_same_event():
    messages = [
        {"reader_agent": 1, "node_id": "a", "cross_agent": True,
         "channel": "teammate_message", "retrieval_step": 1,
         "route_relevant": True},
        {"reader_agent": 1, "node_id": "b", "cross_agent": True,
         "channel": "teammate_message", "retrieval_step": 2,
         "route_relevant": False},
    ]
    result = useful_information_metrics(
        sim(messages=messages, event_map={"a": {"e1"}, "b": {"e1", "e2"}})
    )
    assert result["unique_recipient_node_acquisitions"] == 2
    assert result["unique_recipient_event_acquisitions"] == 2
    assert result["unique_route_relevant_recipient_event_acquisitions"] == 1


def test_self_observed_before_cross_agent_read_is_not_new_acquisition():
    messages = [{
        "reader_agent": 1, "node_id": "n3", "cross_agent": True,
        "channel": "teammate_message", "retrieval_step": 4,
        "route_relevant": True, "new_information": True,
    }]
    discoveries = [{
        "agent": 1, "hazard_id": "n3", "source": "self_observation", "step": 2,
    }]
    result = useful_information_metrics(sim(messages=messages, discoveries=discoveries))
    assert result["unique_recipient_node_acquisitions"] == 1
    assert result["unique_new_recipient_node_acquisitions"] == 0


def test_empty_information_denominators_remain_null():
    result = useful_information_metrics(sim())
    assert result["route_relevant_hazard_read_ratio"] is None
    assert result["unique_acquisition_route_relevance_ratio"] is None
    assert result["unique_new_acquisition_route_relevance_ratio"] is None
    assert result["unique_event_acquisition_route_relevance_ratio"] is None
