import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const res = await fetch('http://127.0.0.1:8000/api/skeletons', { cache: 'no-store' });
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Backend not reachable' }, { status: 500 });
  }
}
