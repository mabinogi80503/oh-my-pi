import { afterEach, describe, expect, it, vi } from "bun:test";
import { AsyncJobManager } from "../../src/async";
import { Settings } from "../../src/config/settings";
import { runEvalWorkpool } from "../../src/eval/workpool-bridge";
import { AgentRegistry } from "../../src/registry/agent-registry";
import * as discovery from "../../src/task/discovery";
import type { AgentDefinition } from "../../src/task/types";
import { WorkPoolRegistry } from "../../src/task/workpool";
import type { ToolSession } from "../../src/tools";

const SCOUT: AgentDefinition = {
	name: "scout",
	description: "Test scout",
	systemPrompt: "Inspect things.",
	source: "bundled",
};

const managers = new Set<AsyncJobManager>();

function makeSession(modelRoles?: Record<string, string>): ToolSession {
	const manager = new AsyncJobManager({ retentionMs: 0 });
	managers.add(manager);
	return {
		cwd: "/tmp",
		hasUI: false,
		settings: Settings.isolated({
			"task.maxConcurrency": 2,
			"task.maxRecursionDepth": 2,
			"task.isolation.enabled": false,
			"task.enableLsp": false,
			...(modelRoles ? { modelRoles } : {}),
		}),
		asyncJobManager: manager,
		getAgentId: () => "Main",
		getSessionFile: () => null,
		getSessionSpawns: () => "*",
		getArtifactsDir: () => null,
	};
}

afterEach(async () => {
	for (const manager of managers) await manager.dispose();
	managers.clear();
	vi.restoreAllMocks();
	AgentRegistry.resetGlobalForTests();
	WorkPoolRegistry.resetForTests();
});

describe("runEvalWorkpool", () => {
	it("validates operation arguments", async () => {
		const session = makeSession();
		await expect(runEvalWorkpool(null, { session })).rejects.toThrow("arguments must be an object");
		await expect(runEvalWorkpool({}, { session })).rejects.toThrow("requires an op");
		await expect(runEvalWorkpool({ op: "create", agent: 4 }, { session })).rejects.toThrow(
			"agent must be a non-empty string",
		);
		await expect(runEvalWorkpool({ op: "status", name: "" }, { session })).rejects.toThrow(
			"requires a non-empty name",
		);
	});

	it("rejects malformed pool model selectors", async () => {
		const session = makeSession();
		await expect(runEvalWorkpool({ op: "create", model: 4 }, { session })).rejects.toThrow("workpool model must be");
		await expect(runEvalWorkpool({ op: "create", model: "  " }, { session })).rejects.toThrow(
			"workpool model must be a non-empty string",
		);
		await expect(runEvalWorkpool({ op: "create", model: [] }, { session })).rejects.toThrow("workpool model must be");
		await expect(runEvalWorkpool({ op: "create", model: ["openai/gpt-4o", " "] }, { session })).rejects.toThrow(
			"workpool model must be",
		);
	});

	it("keeps the raw pool model selector, its role, and per-pool independence", async () => {
		vi.spyOn(discovery, "discoverAgents").mockResolvedValue({ agents: [SCOUT], projectAgentsDir: null });
		const session = makeSession({ reviewer: "openai/gpt-4o", cheap: "openai/gpt-4o-mini" });

		const rolePool = await runEvalWorkpool(
			{ op: "create", agent: "scout", name: "role-pool", model: "@reviewer" },
			{ session },
		);
		const concretePool = await runEvalWorkpool(
			{ op: "create", agent: "scout", name: "concrete-pool", model: ["anthropic/claude-sonnet-4-5", "@cheap"] },
			{ session },
		);
		const plainConcretePool = await runEvalWorkpool(
			{ op: "create", agent: "scout", name: "plain-concrete-pool", model: "anthropic/claude-sonnet-4-5" },
			{ session },
		);
		expect(rolePool).toMatchObject({ name: "role-pool" });
		expect(concretePool).toMatchObject({ name: "concrete-pool" });
		expect(plainConcretePool).toMatchObject({ name: "plain-concrete-pool" });

		const role = WorkPoolRegistry.global().get("Main", "role-pool");
		const concrete = WorkPoolRegistry.global().get("Main", "concrete-pool");
		const plainConcrete = WorkPoolRegistry.global().get("Main", "plain-concrete-pool");
		// The raw selector rides the pool so every new worker's first turn reselects it.
		expect(role?.model).toBe("@reviewer");
		expect(concrete?.model).toEqual(["anthropic/claude-sonnet-4-5", "@cheap"]);
		// A role selector keeps its role identity (and thus its configured fallback chain).
		expect(role?.policy.modelRole).toBe("reviewer");
		expect(role?.policy.modelOverride).toEqual(["openai/gpt-4o"]);
		// A later role alias retains its identity after an earlier concrete candidate.
		expect(concrete?.policy.modelRole).toBe("cheap");
		expect(concrete?.policy.modelOverride).toEqual(["anthropic/claude-sonnet-4-5", "openai/gpt-4o-mini"]);
		// A concrete selector has no role identity, so execution inherits the default fallback chain.
		expect(plainConcrete?.policy.modelRole).toBeUndefined();
		expect(plainConcrete?.policy.modelOverride).toEqual(["anthropic/claude-sonnet-4-5"]);
	});

	it("rejects unknown pool names", async () => {
		const session = makeSession();
		await expect(runEvalWorkpool({ op: "status", name: "missing" }, { session })).rejects.toThrow(
			'unknown workpool "missing"',
		);
	});

	it("creates unique default names and validates push and peek arguments", async () => {
		vi.spyOn(discovery, "discoverAgents").mockResolvedValue({ agents: [SCOUT], projectAgentsDir: null });
		const session = makeSession();
		const events: Array<Record<string, unknown>> = [];
		const first = await runEvalWorkpool(
			{ op: "create", agent: "scout" },
			{ session, emitStatus: event => events.push(event) },
		);
		const second = await runEvalWorkpool({ op: "create", agent: "scout" }, { session });
		expect(first).toEqual({ name: "scout-pool", agent: "scout", limit: 2 });
		expect(second).toEqual({ name: "scout-pool-2", agent: "scout", limit: 2 });
		expect(events).toEqual([{ op: "workpool", action: "create", pool: "scout-pool", count: 2 }]);
		await expect(runEvalWorkpool({ op: "push", name: "scout-pool", items: ["ok", 1] }, { session })).rejects.toThrow(
			"items string array",
		);
		expect(await runEvalWorkpool({ op: "peek", name: "scout-pool" }, { session })).toEqual({
			batches: [],
			pending: 0,
		});
		await expect(runEvalWorkpool({ op: "wait", name: "scout-pool" }, { session })).rejects.toThrow(
			'unknown workpool operation "wait"',
		);
	});
});
