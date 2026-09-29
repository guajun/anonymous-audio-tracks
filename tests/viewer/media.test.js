import test from "node:test";
import assert from "node:assert/strict";

import { createObjectUrlManager } from "../../viewer/js/media.js";

function fakeApi() {
  const created = [];
  const revoked = [];
  let counter = 0;
  return {
    created,
    revoked,
    create(blob) {
      counter += 1;
      const url = `blob:fake-${counter}`;
      created.push({ url, blob });
      return url;
    },
    revoke(url) {
      revoked.push(url);
    },
  };
}

test("replacing a blob revokes the previous URL exactly once", () => {
  const api = fakeApi();
  const manager = createObjectUrlManager(api);
  const first = manager.set({ name: "one" });
  assert.equal(first, "blob:fake-1");
  const second = manager.set({ name: "two" });
  assert.equal(second, "blob:fake-2");
  assert.deepEqual(api.revoked, ["blob:fake-1"]);
  assert.equal(manager.url, "blob:fake-2");
  assert.equal(manager.revokeCount, 1);
});

test("clearing revokes and forgets; a second clear is a no-op", () => {
  const api = fakeApi();
  const manager = createObjectUrlManager(api);
  manager.set({ name: "one" });
  manager.clear();
  manager.clear();
  assert.deepEqual(api.revoked, ["blob:fake-1"]);
  assert.equal(manager.url, null);
  assert.equal(manager.revokeCount, 1);
});

test("set(null) revokes the current URL before switching away", () => {
  const api = fakeApi();
  const manager = createObjectUrlManager(api);
  manager.set({ name: "one" });
  const result = manager.set(null);
  assert.equal(result, null);
  assert.deepEqual(api.revoked, ["blob:fake-1"]);
});

test("events keep an auditable create/revoke trail for the browser check", () => {
  const api = fakeApi();
  const manager = createObjectUrlManager(api);
  manager.set({ name: "one" });
  manager.set({ name: "two" });
  manager.clear();
  assert.deepEqual(
    manager.events.map((event) => event.type),
    ["create", "revoke", "create", "revoke"],
  );
  assert.deepEqual(api.revoked, ["blob:fake-1", "blob:fake-2"]);
});
