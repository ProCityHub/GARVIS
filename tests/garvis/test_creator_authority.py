from garvis.creator_authority import (
    AUTHORITY_SOURCE,
    CREATOR,
    CREATOR_ASSERTION,
    CreatorAction,
    CreatorAuthority,
    require_creator_authority,
)


def test_creator_is_canonical_authority() -> None:
    assert CREATOR == "Adrien D. Thomas"
    assert AUTHORITY_SOURCE == "CREATOR_DIRECTIVE"
    assert CREATOR in CREATOR_ASSERTION


def test_internal_heartbeat_actions_are_standing() -> None:
    authority = CreatorAuthority()
    for action in (
        CreatorAction.RESEARCH,
        CreatorAction.REASON,
        CreatorAction.CAPTURE_PREDICTION_WITNESS,
        CreatorAction.VERIFY,
        CreatorAction.LEARN,
        CreatorAction.CONTINUE_HEARTBEAT,
    ):
        require_creator_authority(authority, action)


def test_merge_and_deploy_are_not_internal_actions() -> None:
    authority = CreatorAuthority()
    assert authority.permits("merge") is False
    assert authority.permits("deploy") is False


def test_creator_authority_hash_shape() -> None:
    assert len(CreatorAuthority().sha256) == 64
