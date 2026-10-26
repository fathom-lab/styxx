// anchor_sdk_repro.mjs: F-1 with clawdbot's plugin SDK in jiti's cache, as the bundled telegram channel plugin puts
// it there in the gateway (it imports `clawdbot/plugin-sdk`, which the loader aliases to dist/plugin-sdk/index.js).
//
//   node anchor_sdk_repro.mjs <dist dir> <plugin dir>     (plugin dir: the fixed W:gate/plugin, with tests/fakes)
//
// From a module jiti evaluates, three ways of reaching deliver.js are read against a native import, before and after
// the SDK goes through jiti: the 0.4.0 way (an import(), kept as tests/fakes/import_deliver_v040.mjs), a require()
// of deliver.js (which misses jiti's copy on Windows only because jiti's require.cache keys use forward slashes), and
// the fix (index.js loadNativeDist: native_import.mjs's import(), the helper loaded through Node's require). The
// runner is initialised natively with one styxx-gate message_sending handler, as the gateway's loader does. Nothing
// is sent. Loading the SDK runs provider discovery code, which may try the network with whatever credentials the
// environment holds; run it where that is acceptable. This is a reproduction in another process, not the gateway.
// Prints one JSON object.
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";

const [DIST, PLUGIN_DIR] = process.argv.slice(2);
const out = { node: process.version };
const creq = createRequire(path.join(DIST, "plugins", "loader.js"));
const { createJiti } = creq("jiti");
out.jiti = creq("jiti/package.json").version;
const hrg = await import(pathToFileURL(path.join(DIST, "plugins", "hook-runner-global.js")).href);
hrg.initializeGlobalHookRunner({ hooks: [], plugins: [], diagnostics: [], typedHooks: [{ hookName: "message_sending", pluginId: "styxx-gate", handler: () => ({}) }] });
const sdk = path.join(DIST, "plugin-sdk", "index.js");
const jiti = createJiti(pathToFileURL(path.join(DIST, "plugins", "loader.js")).href, { interopDefault: true,
    extensions: [".ts", ".tsx", ".mts", ".cts", ".mtsx", ".ctsx", ".js", ".mjs", ".cjs", ".json"], alias: { "clawdbot/plugin-sdk": sdk } });
const oldWay = jiti(path.join(PLUGIN_DIR, "tests", "fakes", "import_deliver_v040.mjs"));
const plugin = jiti(path.join(PLUGIN_DIR, "index.js"));
const nreq = createRequire(path.join(DIST, "entry.js"));
const native = await import(pathToFileURL(path.join(DIST, "infra", "outbound", "deliver.js")).href);

async function readings() {
    const nat = await plugin.loadNativeDist(DIST, PLUGIN_DIR);
    return {
        import_v040_same_as_native: (await oldWay.importDeliver(DIST)).deliverOutboundPayloads === native.deliverOutboundPayloads,
        require_same_as_native: nreq(path.join(DIST, "infra", "outbound", "deliver.js")).deliverOutboundPayloads === native.deliverOutboundPayloads,
        fix_same_as_native: nat.deliverOutboundPayloads === native.deliverOutboundPayloads,
        fix_runner_is_the_gateways: nat.getGlobalHookRunner() === hrg.getGlobalHookRunner(),
        fix_holds_gate: plugin.runnerHoldsGate(nat),
    };
}
out.before_sdk = await readings();
const t0 = Date.now();
jiti(sdk);
out.sdk_through_jiti_ms = Date.now() - t0;
out.after_sdk = await readings();
out.after_sdk.jiti_runner_is_null = jiti(path.join(DIST, "plugins", "hook-runner-global.js")).getGlobalHookRunner() === null;
console.log(JSON.stringify(out));
process.exit(0);
