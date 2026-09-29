// Object-URL lifecycle for user-selected audio files.
//
// The audio element always points at a `blob:` URL that wraps the local File;
// the previous URL is revoked before a new one is created (and again when the
// session is cleared).  No remote URL is ever involved.

/**
 * @param {{create: (blob: Blob) => string, revoke: (url: string) => void}} api
 */
export function createObjectUrlManager(api) {
  let current = null;
  const events = [];

  return {
    /** Replace the current blob URL, revoking the previous one first. */
    set(blob) {
      if (current !== null) {
        api.revoke(current);
        events.push({ type: "revoke", url: current });
        current = null;
      }
      if (blob) {
        current = api.create(blob);
        events.push({ type: "create", url: current });
      }
      return current;
    },
    /** Revoke and forget the current URL (file switched away or load failed). */
    clear() {
      if (current !== null) {
        api.revoke(current);
        events.push({ type: "revoke", url: current });
        current = null;
      }
      return null;
    },
    get url() {
      return current;
    },
    get events() {
      return events.slice();
    },
    get revokeCount() {
      return events.filter((event) => event.type === "revoke").length;
    },
  };
}
