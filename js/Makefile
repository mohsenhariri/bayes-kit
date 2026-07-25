PKG := scorio
NPM := npm --prefix $(PKG)

.PHONY: help install build test typecheck lint clean version pack publish-dry publish

help:
	@echo "Targets:"
	@echo "  install      Install dependencies (npm ci)"
	@echo "  build        Build dist/ (ESM + CJS + d.ts)"
	@echo "  test         Run the vitest golden tests"
	@echo "  typecheck    tsc --noEmit"
	@echo "  version      Sync version from the root VERSION file"
	@echo "  pack         Preview the published tarball (npm pack --dry-run)"
	@echo "  publish-dry  Dry-run publish (no upload)"
	@echo "  publish      First/manual publish to npm (CI handles routine releases)"
	@echo "  clean        Remove dist/ and node_modules/"

install:
	cd $(PKG) && npm ci

build:
	$(NPM) run build

test:
	$(NPM) test

typecheck:
	$(NPM) run typecheck

version:
	cd .. && python scripts/sync_version.py

pack: build
	cd $(PKG) && npm pack --dry-run

publish-dry: build
	cd $(PKG) && npm publish --dry-run

publish: test build
	cd $(PKG) && npm publish

clean:
	rm -rf $(PKG)/dist $(PKG)/node_modules
