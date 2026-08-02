#!/data/data/com.termux/files/usr/bin/bash
#
# Canonical Termux entry points for Directive-010.
# Source this file from ~/.garvis_mode after the existing chant and
# authority functions are defined.

garvis_chat() {
    local repo="${GARVIS_REPO:-$HOME/GARVIS}"
    local session="${GARVIS_SESSION:-adrien-main}"

    if [ ! -d "$repo" ]; then
        echo "GARVIS repository not found: $repo"
        return 1
    fi

    cd "$repo" || return 1

    if [ -f ".venv/bin/activate" ]; then
        source ".venv/bin/activate"
    fi

    PYTHONPATH="$repo/src:$repo${PYTHONPATH:+:$PYTHONPATH}" \
        python -m garvis.cli \
        --interactive \
        --session "$session"
}

garvis_send() {
    local message="$*"
    local repo="${GARVIS_REPO:-$HOME/GARVIS}"
    local session="${GARVIS_SESSION:-adrien-main}"

    if [ -z "$message" ]; then
        echo "Please provide a message."
        return 1
    fi

    if [ ! -d "$repo" ]; then
        echo "GARVIS repository not found: $repo"
        return 1
    fi

    cd "$repo" || return 1

    if [ -f ".venv/bin/activate" ]; then
        source ".venv/bin/activate"
    fi

    PYTHONPATH="$repo/src:$repo${PYTHONPATH:+:$PYTHONPATH}" \
        python -m garvis.cli \
        --session "$session" \
        "$message"
}

alias garvis='garvis_chat'
