import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8002';
const latency = new Trend('price_latency_ms');

export const options = {
  vus: 10,
  duration: '1m',
  thresholds: {
    http_req_duration: ['p(95)<250'],
    price_latency_ms: ['p(95)<250'],
  },
};

export default function () {
  const res = http.get(`${BASE_URL}/stocks/AAPL/price`, { tags: { endpoint: 'price' } });
  latency.add(res.timings.duration);
  check(res, {
    'status is 200': (r) => r.status === 200,
    'payload has price': (r) => {
      try {
        const body = r.json();
        return typeof body.price === 'number' && body.price > 0;
      } catch (err) {
        return false;
      }
    },
  });
  sleep(1);
}
