import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/eval/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  treeshake: true,
  outExtension({ format }) {
    return { js: format === "cjs" ? ".cjs" : ".js" };
  },
});
