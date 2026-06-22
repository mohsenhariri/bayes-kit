# Local developer tasks for the npm package (js/scorio).
#
# Routine releases go through GitHub Actions (.github/workflows/npm-publish.yml),
# triggered by a `js-v*` GitHub Release. Use `make publish` here only for the
# FIRST manual publish (so the package exists and trusted publishing can then be
# configured on npmjs.com), or as an escape hatch.

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

# Propagate the root VERSION file into package.json (and the rest of the repo).
version:
	cd .. && python scripts/sync_version.py

pack: build
	cd $(PKG) && npm pack --dry-run

publish-dry: build
	cd $(PKG) && npm publish --dry-run

# First/manual publish. Requires `npm login` locally. Provenance is only
# available from CI (OIDC), so the manual first publish has none; subsequent
# CI publishes via trusted publishing add provenance automatically.
publish: test build
	cd $(PKG) && npm publish

clean:
	rm -rf $(PKG)/dist $(PKG)/node_modules
