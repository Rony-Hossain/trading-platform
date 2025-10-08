import ws from 'k6/ws';
import { check } from 'k6';

const WS_URL = __ENV.WS_URL || 'ws://localhost:8002/ws/AAPL';

export const options = {
  scenarios: {
    websocket_load: {
      executor: 'shared-iterations',
      vus: 100,
      iterations: 100,
      maxDuration: '5m',
    },
  },
  thresholds: {
    'checks{type:first_tick}': ['rate>0.99'],
  },
};

export default function () {
  const res = ws.connect(WS_URL, {}, function (socket) {
    socket.setTimeout(function () {
      socket.close();
    }, 10000);

    socket.on('message', function (message) {
      try {
        const payload = JSON.parse(message);
        check(payload, {
          'price present': (data) => typeof data.price === 'number' && data.price > 0,
          'timestamp present': (data) => typeof data.ts === 'string' || typeof data.timestamp === 'string',
        }, { type: 'first_tick' });
        socket.close();
      } catch (err) {
        socket.close();
      }
    });

    socket.on('error', function () {
      socket.close();
    });
  });

  check(res, { 'connected': (r) => r && r.status === 101 });
}
