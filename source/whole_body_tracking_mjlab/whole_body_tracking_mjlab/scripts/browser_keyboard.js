function installKeyboard(config) {
  // Reconnects replace listeners and held keys instead of accumulating handlers.
  window.sproutKeyboardCleanup?.();
  const controller = new AbortController();
  const options = {capture: true, signal: controller.signal};
  const keys = new Set();
  const movement = new Set(['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);
  const shortcuts = {KeyP: 'toggle_pause', KeyR: 'reset', Space: 'stop'};
  const url = new URL(window.location.href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.port = String(config.port);
  url.pathname = '/';
  url.search = '';
  url.hash = '';
  let socket;
  let ready = false;
  let disposed = false;
  let retry;
  const status = document.createElement('div');
  status.textContent = 'Connecting keyboard controls…';
  Object.assign(status.style, {position: 'fixed', bottom: '12px', left: '12px', zIndex: '9999',
    background: '#20252b', color: 'white', padding: '8px 12px', borderRadius: '6px', pointerEvents: 'none'});
  document.body.append(status);

  function send(message) {
    if (ready && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  }
  function release() {
    if (keys.size) send({action: 'release'});
    keys.clear();
  }
  function drive() {
    const down = (...codes) => Number(codes.some(code => keys.has(code)));
    send({action: 'drive', forward: down('KeyW', 'ArrowUp') - down('KeyS', 'ArrowDown'),
      lateral: down('KeyA', 'ArrowLeft') - down('KeyD', 'ArrowRight'),
      turn: down('KeyQ') - down('KeyE')});
  }
  function editable(target) {
    return target instanceof Element && !!target.closest('input, textarea, select, [contenteditable="true"], [role="textbox"]');
  }
  function connect() {
    socket = new WebSocket(url);
    socket.onopen = () => socket.send(JSON.stringify({client: config.client, token: config.token}));
    socket.onmessage = event => {
      ready = JSON.parse(event.data).ready === true;
      status.hidden = ready;
    };
    socket.onclose = () => {
      ready = false;
      keys.clear();
      status.hidden = false;
      status.textContent = `Keyboard disconnected — reconnecting on port ${config.port}…`;
      if (!disposed) retry = setTimeout(connect, 1000);
    };
  }
  window.addEventListener('keydown', event => {
    if (editable(event.target)) {
      release();
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      release();
      if (movement.has(event.code)) event.stopImmediatePropagation();
      return;
    }
    if (!movement.has(event.code) && !(event.code in shortcuts)) return;
    // Capture before Viser's document handlers, so robot keys never move its camera.
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.repeat || !ready) return;
    if (movement.has(event.code)) {
      keys.add(event.code);
      drive();
    } else if (event.code in shortcuts) {
      release();
      send({action: shortcuts[event.code]});
    }
  }, options);
  window.addEventListener('keyup', event => {
    if (!movement.has(event.code)) return;
    const wasHeld = keys.delete(event.code);
    if (!editable(event.target)) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
    if (wasHeld) {
      if (keys.size) drive();
      else send({action: 'release'});
    }
  }, options);
  window.addEventListener('blur', release, options);
  window.addEventListener('pagehide', release, options);
  document.addEventListener('visibilitychange', () => {if (document.hidden) release();}, options);
  document.addEventListener('focusin', event => {if (editable(event.target)) release();}, options);
  // Stop held keyboard motion when switching to sidebar controls, including Pause/Reset.
  document.addEventListener('pointerdown', event => {
    if (!(event.target instanceof HTMLCanvasElement)) release();
  }, options);
  // A fixed chase camera has no mouse bindings or pointer lock. Swallow viewer
  // orbit/pan/zoom events on the scene while preserving all sidebar interactions.
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'mousedown', 'wheel', 'contextmenu', 'dblclick']) {
    window.addEventListener(type, event => {
      if (!(event.target instanceof HTMLCanvasElement)) return;
      if (type === 'pointerdown' && document.activeElement instanceof HTMLElement) document.activeElement.blur();
      event.preventDefault();
      event.stopImmediatePropagation();
    }, {...options, passive: false});
  }
  // The server expires a held command if heartbeats cease (lost focus/network/tab suspension).
  const heartbeat = setInterval(() => {if (keys.size) drive();}, 150);
  window.sproutKeyboardCleanup = () => {
    release();
    disposed = true;
    controller.abort();
    clearInterval(heartbeat);
    clearTimeout(retry);
    socket?.close();
    status.remove();
  };
  connect();
}
