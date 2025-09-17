const LOG_PREFIX = '[facilitator-channel]';

let facilitatorChannel = null;
let manifestString = null;
let manifestRevision = 0;
let lastSentRevision = 0;
let needsReplay = false;
let lastManifestTimestamp = null;

function detachChannel() {
  if (!facilitatorChannel) {
    return;
  }
  facilitatorChannel.removeEventListener('open', handleChannelOpen);
  facilitatorChannel.removeEventListener('close', handleChannelClose);
  facilitatorChannel.removeEventListener('error', handleChannelError);
  facilitatorChannel = null;
}

function handleChannelOpen() {
  console.info(`${LOG_PREFIX} data channel open`);
  if (manifestString) {
    needsReplay = true;
    sendManifest('channel-open');
  }
}

function handleChannelClose() {
  console.warn(`${LOG_PREFIX} data channel closed`);
  if (manifestString) {
    needsReplay = true;
  }
}

function handleChannelError(event) {
  console.error(`${LOG_PREFIX} data channel error`, event);
  if (manifestString) {
    needsReplay = true;
  }
}

function sendManifest(reason) {
  if (!facilitatorChannel) {
    console.debug(`${LOG_PREFIX} skip manifest send (${reason}) – no channel`);
    return false;
  }
  if (facilitatorChannel.readyState !== 'open') {
    console.debug(
      `${LOG_PREFIX} skip manifest send (${reason}) – channel state ${facilitatorChannel.readyState}`,
    );
    return false;
  }
  if (!manifestString) {
    console.debug(`${LOG_PREFIX} skip manifest send (${reason}) – no cached manifest`);
    return false;
  }

  if (!needsReplay && lastSentRevision === manifestRevision) {
    console.debug(
      `${LOG_PREFIX} skip manifest send (${reason}) – revision ${manifestRevision} already delivered`,
    );
    return false;
  }

  try {
    facilitatorChannel.send(manifestString);
    lastSentRevision = manifestRevision;
    needsReplay = false;
    console.info(
      `${LOG_PREFIX} manifest revision ${manifestRevision} sent (${reason})`,
    );
    return true;
  } catch (err) {
    console.error(`${LOG_PREFIX} failed to send manifest (${reason})`, err);
    needsReplay = true;
    return false;
  }
}

export function registerFacilitatorDataChannel(channel) {
  if (facilitatorChannel === channel) {
    return;
  }

  detachChannel();

  if (!channel) {
    console.info(`${LOG_PREFIX} cleared data channel`);
    needsReplay = !!manifestString;
    return;
  }

  facilitatorChannel = channel;
  facilitatorChannel.addEventListener('open', handleChannelOpen);
  facilitatorChannel.addEventListener('close', handleChannelClose);
  facilitatorChannel.addEventListener('error', handleChannelError);

  if (facilitatorChannel.readyState === 'open') {
    needsReplay = true;
    sendManifest('channel-ready');
  } else if (manifestString) {
    needsReplay = true;
  }
}

export function sendFacilitatorManifest(manifest) {
  if (manifest === undefined || manifest === null) {
    console.info(`${LOG_PREFIX} clearing cached manifest`);
    manifestString = null;
    needsReplay = false;
    manifestRevision += 1;
    return false;
  }

  try {
    manifestString = typeof manifest === 'string' ? manifest : JSON.stringify(manifest);
  } catch (err) {
    console.error(`${LOG_PREFIX} unable to serialise manifest`, err);
    return false;
  }

  manifestRevision += 1;
  lastManifestTimestamp = Date.now();
  needsReplay = true;
  return sendManifest('manifest-update');
}

export function resendFacilitatorManifest() {
  if (!manifestString) {
    console.warn(`${LOG_PREFIX} no manifest cached to resend`);
    return false;
  }
  needsReplay = true;
  return sendManifest('manual-resend');
}

export function getFacilitatorManifestState() {
  return {
    hasManifest: !!manifestString,
    revision: manifestRevision,
    lastSentRevision,
    needsReplay,
    channelReadyState: facilitatorChannel ? facilitatorChannel.readyState : 'closed',
    bufferedAmount: facilitatorChannel ? facilitatorChannel.bufferedAmount : 0,
    lastManifestTimestamp,
  };
}

export function clearFacilitatorManifestCache() {
  manifestString = null;
  needsReplay = false;
  lastSentRevision = 0;
  manifestRevision = 0;
  lastManifestTimestamp = null;
}

function installGlobalHelpers() {
  if (typeof window === 'undefined') {
    return;
  }
  const api = {
    setDataChannel: registerFacilitatorDataChannel,
    sendManifest: sendFacilitatorManifest,
    resendManifest: resendFacilitatorManifest,
    getState: getFacilitatorManifestState,
    clearCache: clearFacilitatorManifestCache,
  };
  if (!window.facilitatorChannel) {
    window.facilitatorChannel = api;
  } else {
    Object.assign(window.facilitatorChannel, api);
  }
}

installGlobalHelpers();

export function __getFacilitatorChannelForTesting() {
  return {
    channel: facilitatorChannel,
    manifestString,
    manifestRevision,
    lastSentRevision,
    needsReplay,
    lastManifestTimestamp,
  };
}
