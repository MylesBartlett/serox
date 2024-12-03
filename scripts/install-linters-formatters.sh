#!/usr/bin/sh
if ! command -v taplo &>/dev/null; then
	echo "Installing taplo using cargo"
	cargo install taplo-cli --locked --version 0.9.0
fi

if ! command -v dprint &>/dev/null; then
	echo "Installing dprint using cargo"
	cargo install dprint --locked
fi

if ! command -v yamlfmt &>/dev/null; then
	echo "Installing yamlfmt using go"
	go install github.com/google/yamlfmt/cmd/yamlfmt@latest
fi
