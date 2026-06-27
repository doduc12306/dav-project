import { NextResponse } from 'next/server';

export async function GET(request: Request, { params }: { params: Promise<{ filename: string }> }) {
  try {
    const { filename } = await params;
    const res = await fetch(`http://127.0.0.1:8000/api/predict/${filename}`, { cache: 'no-store' });
    if (!res.ok) {
      return NextResponse.json({ error: 'Backend error' }, { status: res.status });
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Backend not reachable' }, { status: 500 });
  }
}
