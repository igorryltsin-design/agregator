PACKAGE_VERSION ?= $(shell git describe --tags --always --dirty)
DIST_DIR := dist

.PHONY: package clean lint

package:
	@./scripts/package_release.sh $(PACKAGE_VERSION)

clean:
	rm -rf $(DIST_DIR)

lint:
	@echo "TODO: add linters (flake8, eslint) as needed"
