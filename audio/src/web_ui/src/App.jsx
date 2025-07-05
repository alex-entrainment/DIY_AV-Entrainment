import React, { useEffect } from 'react';
import { Button } from './components/ui/button';
import {
  start,
  stop,
  pause,
  resume,
  seek,
  sendUpdate,
  toggleGpu,
  loadTrackFromServer,
  loadNoiseFromServer,
  addClipFromServer,
  initSelects,
} from './main.js';

export default function App() {
  useEffect(() => {
    initSelects();
  }, []);

  return (
    <div className="p-4 space-y-4 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Realtime Backend Web Demo</h1>
      <div className="space-y-2">
        <input type="file" id="json-upload" accept=".json" className="block" />
        <div className="flex space-x-2">
          <select id="track-select" className="flex-1 bg-gray-800 p-2 rounded" />
          <Button id="load-track" onClick={loadTrackFromServer}>Load Track</Button>
        </div>
        <input type="file" id="noise-upload" accept=".noise" className="block" />
        <div className="flex space-x-2">
          <select id="noise-select" className="flex-1 bg-gray-800 p-2 rounded" />
          <Button id="load-noise" onClick={loadNoiseFromServer}>Insert Noise</Button>
        </div>
        <input type="file" id="clip-upload" accept=".wav,.flac,.mp3" multiple className="block" />
        <div className="flex space-x-2">
          <select id="clip-select" multiple className="flex-1 bg-gray-800 p-2 rounded" />
          <Button id="add-clip" onClick={addClipFromServer}>Add Clip</Button>
        </div>
        <textarea id="track-json" rows="10" cols="80" className="w-full bg-gray-800 p-2 rounded" defaultValue={`{\n  "global": {"sample_rate": 44100},\n  "progression": [],\n  "background_noise": {},\n  "overlay_clips": []\n}`} />
        <label className="block">Start time (s): <input id="start-time" type="number" step="0.1" defaultValue="0" className="ml-2 text-black" /></label>
        <label className="block"><input type="checkbox" id="gpu-enable" className="mr-2" /> Enable GPU</label>
        <div className="space-x-2">
          <Button id="start" onClick={start}>Start</Button>
          <Button id="pause" onClick={pause}>Pause</Button>
          <Button id="resume" onClick={resume}>Resume</Button>
          <Button id="stop" onClick={stop}>Stop</Button>
        </div>
        <div className="flex items-center space-x-2">
          <label>Seek to (s): <input id="seek-time" type="number" step="0.1" defaultValue="0" className="ml-2 text-black" /></label>
          <Button id="seek-button" onClick={seek}>Seek</Button>
          <Button id="update-track" onClick={sendUpdate}>Update Track</Button>
        </div>
        <div>Current step: <span id="current-step">0</span></div>
        <div>Elapsed samples: <span id="elapsed-samples">0</span></div>
      </div>
    </div>
  );
}
