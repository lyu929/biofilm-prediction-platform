// node-worker-wrapper.js
import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export function createArtistooWorker() {
    const worker = new Worker(join(__dirname, 'artistoo-worker.js'));
    
    // Inject CPM into worker context
    worker.postMessage({
        command: 'load-artistoo',
        data: { CPM: CPM } // Pass Artistoo CPM class
    });
    
    return worker;
}
