#define WP 32


#define BM (WP-1)
#define dt unsigned short
#define dts 2

#define TGTG  1
#define TGNTG 0

#define LBMAX 0xffff
#define LB2MAX 0xffffffff

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#define W 1024
#define HWW 9
#define rep 2

#define HW (W/2)
#define QW (W/4)
#define HWP (WP/2)
#define QWP (WP/4)
#define EWP (WP/8)
#define XWP (WP/16)

#define TN (W/(rep*2))
#define WN (TN/WP)
#define MASKAD (~(-1 << HWW))

__device__ __inline__ static void detectRuns1024(ushort* pt, uchar* ct, uchar* ps, uchar* px, ushort* tbf, uint cp2)
{
	uint* ps4 = (uint*)ps;
	uint* pt2 = (uint*)pt;
	ushort* ct2 = (ushort*)ct;
	uint cp = cp2 >> 1;//threadIdx.x + threadIdx.y * WP;
//	uint cp2 = cp << 1;

	uint t0, t1, t2, t3, t4, t5;
	t0 = ps4[cp];
	t0 = __byte_perm(t0, 0, 0x3120);
	t0 &= 0x1010101;
	t0 |= t0 >> 15;
	t0 |= t0 >> 6;
	t0 &= 0xf;
	t1 = px[cp];
	t5 = 0;
	if((t1 & 0x3))// && (t0 & 0x3)
		t5 = 0x1;
	if((t1 & 0xc))// && (t0 & 0xc)
		t5 |= 0x100;
	ct2[cp] = t5;// | ((ct2[cp] & 0xf0f) << 4);

	t1 = t1 | t0;
	px[cp] = t0 | (px[cp] << 4);
	
	t3 = W;
	t4 = W;
	t5 = 0;
	if(t1 & 0x3)
		t3 = cp2;
	if(t1 & 0xc)
		t4 = cp2 + 1;
	if((t1 & 0x6) == 0x6)
		t4 = t3;
	
	t0 = W;
	if(t1 & 0x8)
		t0 = t4;
	pt2[cp] = t0;
	t0 = __ballot_sync(0xffffffff, (t1 & 0xf) != 0xf);
	t0 <<= (WP - threadIdx.x);
	t0 = __clz(t0);
	t5 = (t0 == WP) && ((t1 & 0xf) == 0xf);
	t0 = max((int)0, ((int)threadIdx.x) - 1 - ((int)t0));
	t2 = pt2[threadIdx.y * WP + t0];
	if(t2 == W)
		t2 = (t0 + 1+ threadIdx.y * WP) * 2;
//	__syncthreads();
	
	if(threadIdx.x != 0 && (t1 & 0x1)){
		if(t3 == cp2)
			t3 = t2;
		if(t4 == cp2)
			t4 = t2;
	}
	if(threadIdx.x == (WP-1)){
		t5 = t5 ? t4: t4 + HW ;
		t5 = (t1 & 0x8) ? t5 : (W + HW);
		tbf[threadIdx.y] = t5;
	}
	__syncthreads();

	if(threadIdx.y != 0){
		t5 = tbf[threadIdx.x];
		t0 = __ballot_sync(0xffffffff, (t5 & HW) != 0);
		t0 <<= (WP- threadIdx.y);
		t0 = __clz(t0);
		t0 = max((int)0, ((int)threadIdx.y) - 1 - ((int)t0));
		t1 = tbf[t0];
		if(t1 & W)
			t1 = (t0 + 1)* WP * 2;
		
		t1 = t1 & MASKAD; //remove tags
		cp2 = threadIdx.y*WP*2;
		if(px[cp2>>1] & 0x11){
			if(t3 == cp2)
				t3 = t1;
			if(t4 == cp2)
				t4 = t1;
		}
	}
	t4 <<= 16;
	t4 |= t3;
	pt2[cp] = t4;
}

__global__ static void pass1_1024(ushort* pt, uchar* ps, ushort* b,  ushort* b2, uint h, ushort* eb)
{
	__shared__ __align__(4) ushort lk0[HW];	//upper link
	__shared__ __align__(4) ushort lk1[HW];	//back up of current link
//	__shared__ __align__(4) ushort lk2[HW];	//new value of current link
	__shared__ __align__(4) ushort lk3[HW];	//down linked last block
	__shared__ __align__(4) ushort lk4[HW];	//botton link
	__shared__ uchar ct[HW];	//connection tag of each block
	__shared__ uchar px[QW];	//value of four pixels
	__shared__ ushort tbf[WP];

	uint blockIdxx = blockIdx.x;
	uint cp = (threadIdx.x + threadIdx.y * WP) << 1;
	
//	if(blockIdxx != 3)
//		return;

	ps = ps + blockIdxx * h * W;
	pt = pt + blockIdxx * h * HW / 2;
	eb = eb + blockIdxx * HW * 9;
	h += 2; //calucate 2 more lines

	px[cp>>1] = 0;
//	((ushort*)ct)[cp>>1] = 0;
	detectRuns1024(lk0, ct, ps, px, tbf, cp);
	ps += W;
/*
		{//no error
			//cp = threadIdx.x + threadIdx.y*WP;
			((uint*)(eb))[cp>>1]      = 0;
			((uint*)(eb+HW))[cp>>1]   = 0;
			((uint*)(eb+2*HW))[cp>>1] = 0;
			((uint*)(eb+3*HW))[cp>>1] = 0;
			((uint*)(eb+4*HW))[cp>>1] = 0;
			((uint*)(eb+5*HW))[cp>>1] = 0;
			((uint*)(eb+6*HW))[cp>>1] = 0;
			((uint*)(eb+7*HW))[cp>>1] = 0;
			((uint*)(eb+8*HW))[cp>>1] = 0;
		}
*/
	for(int hh = 1; hh < h; hh++){
		detectRuns1024(lk1, ct, ps, px, tbf, cp);
		ps += W;
		uint lt0 = ((uint*)lk0)[cp>>1];	//backup lk0
		((uint*)lk0)[cp>>1] = lt0 | ((HW << 16) | HW);//((uint*)lk0)[cp>>1];//(W << 16) | W;//initialize lk2
		__syncthreads();
		uint lt1 = ((uint*)lk1)[cp>>1]; //back up lk1, because lk1 will modified next

		{
			ushort t0;
			
			t0 = lt0;
			if(ct[cp]){// every one write lk0 && lk1[lk0[cp]] > lk1[cp]
				lk0[t0] = lt1;
			}
			
			cp++;
			t0 = lt0 >> 16;
			if(ct[cp]){
				lk0[t0] = lt1 >> 16;
			}

			cp--;
			__syncthreads();
		}

		do{
			ushort t0, t1, t2;
			bool changed = 0;

			t1 = lt1;
			t0 = lt0;
			t2 = t1;
			if(ct[cp]){
				t2 = lk0[t0];
				if(t1 != t2)
					changed = 1;
				if(t2 < t1)//update self
					lk1[t1] = t2;
			}

			cp++;
			t1 = lt1 >> 16;
			t0 = lt0 >> 16;
			t2 = t1;
			if(ct[cp]){
				t2 = lk0[t0];
				if(t1 != t2)
					changed = 1;
				if(t2 < t1)
					lk1[t1] = t2;
			}
			cp--;

			changed = __syncthreads_or(changed);
			t1 = lt1;
			if(t1 < HW)
				t1 = lk1[t1];
			lt1 =  __byte_perm(lt1, t1, 0x3254);
			t1 = lt1 >> 16;
			if(t1 < HW)
				t1 = lk1[t1];
			lt1 =  __byte_perm(lt1, t1, 0x5410);

			if(!changed)
				break;
			
			t1 = lt1;
			t0 = lt0;
			if(ct[cp] && (lk0[t0] > t1)){// min write
				lk0[t0] = t1;
			}
			
			cp++;
			t1 = lt1 >> 16;
			t0 = lt0 >> 16;
			if(ct[cp] && (lk0[t0] > t1)){
				lk0[t0] = t1;
			}

			cp--;
			__syncthreads();
		}while(1);
		
		((uint*)lk1)[cp>>1] = lt1;
		
		if((hh & 0x1) && (hh != 1))
		{//only odd line write out
			//resolve linkage by lk3, lk4
			ushort t0, t1, t2;
			
			t0 = lk3[cp];
			t2 = t0;
			if(t0 >= HW && t0 < W)
				t2 = lk3[t0 - HW];
			if(t2 < HW){
				t2 = lk0[t2];
				if(t2 < HW)
					t0 = t2;
				else if(t0 < HW)
					t0 = cp | HW;
			}
			cp++;

			t1 = lk3[cp];
			t2 = t1;
			if(t1 >= HW && t1 < W)
				t2 = lk3[t1 - HW];
			if(t2 < HW){
				t2 = lk0[t2];
				if(t2 < HW)
					t1 = t2;
				else if(t1 < HW)
					t1 = cp | HW;
			}
			cp--;
			
			uint t3 = (t1 << 16) | t0;
				
			((uint*)lk3)[cp>>1] = t3;//write back, next iter, lk4 will use it
			//((uint*)pt)[cp>>1] = t3;//((uint*)lk3)[cp>>1]((uint*)lk3)[cp>>1]
			//pt += HW;
		}
		else
			((uint*)lk3)[cp>>1] = ((uint*)lk0)[cp>>1];//t2;		if((hh & 0x1) == 0)

		__syncthreads();
		((uint*)lk0)[cp>>1] = ((uint*)lk1)[cp>>1];
		if(hh < 4)
			((uint*)lk4)[cp>>1] = ((uint*)lk3)[cp>>1];//t2;
		else if((hh & 0x1)){// && (hh != 1)
			ushort t0, t1;

			t0 = lk4[cp];
			t1 = t0;
			if(t0 < HW){
				t0 = lk3[t0];
				if(t0 >= HW && t0 < W){
					lk3[t0 - HW] = 0x8000 | cp;
					t0 = t1 | 0x8000;
				}
				else if(t0 & 0x8000)
					t0 = t1 | 0x8000;
			}
			lk4[cp] = t0;
			cp++;

			t0 = lk4[cp];
			t1 = t0;
			if(t0 < HW){
				t0 = lk3[t0];
				if(t0 >= HW && t0 < W){
					lk3[t0 - HW] = 0x8000 | cp;
					t0 = t1 | 0x8000;
				}
				else if(t0 & 0x8000)
					t0 = t1 | 0x8000;
			}
			lk4[cp] = t0;
			cp--;
		}

		__syncthreads();
		if((hh & 0x1) && (hh != 1)){
			ushort t0;
			((uint*)pt)[cp>>1] = ((uint*)lk3)[cp>>1];
			pt += HW;

			t0 = lk4[cp];
			if(t0 & 0x8000)
				t0 = (lk3[t0 & MASKAD] & MASKAD) | HW;
			lk4[cp] = t0;
			cp++;

			t0 = lk4[cp];
			if(t0 & 0x8000)
				t0 = (lk3[t0 & MASKAD] & MASKAD) | HW;
			lk4[cp] = t0;
			cp--;
		}
	}

	//write out info for pass2
	{
		b += blockIdxx * 2 * HW;
		b2 += blockIdxx * 2 * HW;

		((uint*)b)[cp>>1] = ((uint*)lk4)[cp>>1];
//		((uint*)b2)[cp>>1] = ((uint*)lk4)[cp>>1];
		b += HW;
		b2 += HW;

		((uint*)b)[cp>>1] = ((uint*)lk1)[cp>>1];
//		((uint*)b2)[cp>>1] = ((uint*)lk1)[cp>>1];
	}
//	if(threadIdx.x == 0 && threadIdx.y == 0)
//		printf("%d end\n", blockIdx.x);
}

__global__ static void pass2_1024(ushort* ib, uint* glabel, uint h, uint pitch)
{
	__shared__ uint   lk0[HW];	//last tag
	__shared__ ushort lk1[HW];	//current link
	__shared__ uint   lk2[HW];	//current tag
	__shared__ ushort lk3[HW];	//bottom flattened link to current
	__shared__ uint labels;

	uint cp = threadIdx.x;

	ushort* b = ib + (h*(pitch+1)*blockIdx.x + 1) * HW;
	{
	ushort tmp = b[cp];
	lk0[cp] = tmp;
	lk3[cp] = tmp;
	lk2[cp] = W;
	b+=HW*pitch;
	}
	labels = 0;
	__syncthreads();

	for(int i = 1;i < h; i++){
		ushort tmp, tmp1;
		//load a new block info
		tmp = b[cp];//the  upper line connection info to bottom line
//		lk1[cp] = tmp;
		if(i>1){
			tmp1 = lk3[cp];
			//update
			if(tmp1 < HW)
				tmp1 = tmp;
			lk3[cp] = tmp1;
		}
		b+=HW;
		//lk2[cp] = b[cp];//the bottom line tags, this proc try to unify this line
		tmp1 = b[cp];
		lk2[cp] = tmp1 + HW;//tmp1 < HW ? tmp1 + HW : tmp1;
//		b+=HW*pitch;
		__syncthreads();
		
		ushort lt0 = lk0[cp];
		if(tmp < HW)//every one write
			lk2[tmp] = lt0;
		else if(tmp < W)
			lk1[tmp - HW] = lt0;
		__syncthreads();

		do{
			bool changed = 0;
			if(tmp < HW){
				changed = lk2[tmp] != lt0;
				if(lk2[tmp] < lt0)
					lk0[lt0] = lk2[tmp];
			}
			else if(tmp < W){
				changed = lk1[tmp - HW] != lt0;
				if(lk1[tmp - HW] < lt0)
					lk0[lt0] = lk1[tmp - HW];
			}
			changed = __syncthreads_or(changed);
			if(lt0 < HW)
				lt0 = lk0[lt0];
			if(!changed)
				break;
			if(tmp < HW){
				if(lk2[tmp] > lt0)
					lk2[tmp] = lt0;
			}
			else if(tmp < W){
				if(lt0 < lk1[tmp - HW])
					lk1[tmp - HW] = lt0;
			}
			__syncthreads();
		}while(1);

		//write out lk2 info, ???link back to bottom?
		b -= HW*(pitch+1);
		b[cp] = lt0;
		b += HW*(pitch+pitch+1);

		tmp = lk2[cp];
		if(tmp < HW)// && tmp < W)// && lk2[cp] == cp,  if dateset correct, this is not necessary
			lk0[tmp] = cp;
			//atomicMin(lk0+tmp, cp);
		__syncthreads();
		//head link together
		tmp = lk2[cp];
		if(tmp < HW)
			tmp = lk0[tmp];
		else if(tmp < W)
			tmp = tmp - HW;
		lk2[cp] = tmp;

		__syncthreads();
		//all leaf in lk2 updates
		tmp = lk2[cp];
		if(tmp < HW)
			tmp = lk2[tmp];

		lk0[cp] = tmp;//so lk0 contains the last block line tags
//		__syncthreads();
		
	}
	b -= HW*pitch;
	b[cp] = lk0[cp];

	b = ib + (h*(pitch+1)*(blockIdx.x+1) -1) * HW;
	//back ward tag assign
	{
	ushort tmp = b[cp], res = LBMAX;
	if(tmp == cp)
		res = atomicInc(&labels, 0x8000);//(h-1) << 12;
	lk2[cp] = res;
	__syncthreads();
	if(tmp < HW)
		res = lk2[tmp];
	lk2[cp] = res;	//last line tags
	b[cp] = res;
	b -= HW;
	}

	__syncthreads();

	for(int i = h-2;i >= 0; i--){
		ushort tmp, tmp2;//, tmp4;
		tmp = b[cp];	//next link info
		//lk1[cp] = tmp;
		b -= HW*pitch;
		tmp2 = b[cp];//current link info
		lk1[cp] = tmp2;
		lk0[cp] = LBMAX;//final tags
		__syncthreads();

/*		tmp4 = W;
		if(tmp >= HW && tmp < W){//race to resolve linked by current block
			tmp4 = tmp - HW;
			lk1[tmp4] = tmp2;
		}
		__syncthreads();

		do{
			bool changed = 0;
			if(tmp4 < HW){
				changed = lk1[tmp4] != tmp2;
				if(tmp2 > lk1[tmp4])
					lk1[tmp2] = lk1[tmp4];
			}
			changed = __syncthreads_or(changed);
			if(tmp2 < HW)
				tmp2 = lk1[tmp2];
			if(!changed)
				break;

			if(tmp4 < HW && tmp2 < lk1[tmp4])
				lk1[tmp4] = tmp2;
			__syncthreads();
		}while(1);
*/
		if(tmp < HW){//next linked
			if(tmp2 < HW)
				lk0[tmp2] = lk2[tmp];
//			else
//				tmp2 = W;
		}

		__syncthreads();
		if(tmp2 == cp && lk0[cp] == LBMAX){//current linked
			lk0[cp] = atomicInc(&labels, 0x8000);//((h-1) << 12) - HW;
		}

		__syncthreads();
		tmp = LBMAX;
		if(tmp2 < HW)
			tmp = lk0[tmp2];

		//write out tags
		b[cp] = tmp;
		b -= HW;
		//switch buffer
		lk2[cp] = tmp;
		__syncthreads();
	}
	//update first line link info
//	b = ib + (h*(pitch+1)*blockIdx.x) * HW;
//	b[cp] = lk3[cp];

	*glabel = labels;
}


__global__ void static pass3_1024(ushort* pt, ushort* ps, ushort* b, uint* label, uint h)
{
	__shared__ ushort cur[W/2];	//last tag
	__shared__ ushort lst[W/2];	//current link
	__shared__ ushort bot[W/2];	//current tag
	__shared__ ushort lnk[W/2]; //current link to last
//	__shared__ uint llabel;

	uint cp = (threadIdx.x + threadIdx.y * WP) << 1;

	pt = pt + (blockIdx.x * h + h - 1) * HW;
	ps = ps + (blockIdx.x * h + h - 1) * HW;

	if(blockIdx.x == 0){
		((uint*)bot)[cp>>1] = LB2MAX;
		b+= HW;
	}else{
		b += (blockIdx.x*2 - 1)*HW;
		((uint*)bot)[cp>>1] = ((uint*)b)[cp>>1];
		b+= 2*HW;
	}
	((uint*)cur)[cp>>1] = ((uint*)b)[cp>>1];
	__syncthreads();

	for(int i = h-1; i>=0; i--){
		//load current link
		((uint*)lnk)[cp>>1] = ((uint*)ps)[cp>>1];
		ps -= HW;
		//switch cur last
		((uint*)lst)[cp>>1] = ((uint*)cur)[cp>>1];
		//clear cur
		((uint*)cur)[cp>>1] = LB2MAX;
		
//		llabel = 0;
		__syncthreads();
		if(lnk[cp] < HW){//link to last
			cur[cp] = lst[lnk[cp]];
		}
		else if(lnk[cp] < W && lnk[cp] == (HW+cp)){//link to local, and a head, assigning new label
			if(blockIdx.x != 0 && i == 0)
				cur[cp] = bot[cp];
			else
				cur[cp] = atomicInc(label, LBMAX);
		}
		if(lnk[cp] & 0x8000){//link to bottom
			if(blockIdx.x == 0)
				cur[cp] = atomicInc(label, LBMAX);
			else
				cur[cp] = bot[lnk[cp] & MASKAD];
		}

		cp ++;
		if(lnk[cp] < HW){//link to last
			cur[cp] = lst[lnk[cp]];
		}
		else if(lnk[cp] < W && lnk[cp] == (HW+cp)){//link to local, and a head, assigning new label
			if(blockIdx.x != 0 && i == 0)
				cur[cp] = bot[cp];
			else
				cur[cp] = atomicInc(label, LBMAX);
		}
		if(lnk[cp] & 0x8000){//link to bottom
			if(blockIdx.x == 0)
				cur[cp] = atomicInc(label, LBMAX);
			else
				cur[cp] = bot[lnk[cp] & MASKAD];
		}
		cp--;
		__syncthreads();
		if(lnk[cp] >= HW && lnk[cp] < W)
			cur[cp] = cur[lnk[cp] - HW];
		cp ++;
		if(lnk[cp] >= HW && lnk[cp] < W)
			cur[cp] = cur[lnk[cp] - HW];
		cp--;

		//write out
		((uint*)pt)[cp>>1] = ((uint*)cur)[cp>>1];
		pt -= HW;
		__syncthreads();
	}

}

void chen_label_1024(uchar* cbpt, uchar* cbpt2, uchar* cbps, uchar* cbb, uchar* cbb2, uchar* cbglabel, uint h, uint bn, uchar* cbeb)
{
	ushort* pt = (ushort*)cbpt;
	ushort* pt2 = (ushort*)cbpt2;
	uchar* ps = (uchar*)cbps;
	ushort* b = (ushort*)cbb;
	ushort* b2 = (ushort*)cbb2;
	uint* glabel = (uint*)cbglabel;
	ushort* eb = (ushort*)cbeb;

	dim3 threads(WP, TN/WP, 1);
    dim3 grid(bn, 1, 1);

	dim3 threads2(HW, 1, 1);
    dim3 grid2(1, 1, 1);

	pass1_1024<<<grid, threads>>>(pt2, ps, b, b2, (h-2)/bn, eb);
	pass2_1024<<<grid2, threads2>>>(b, glabel, bn, 1);
	pass3_1024<<<grid, threads>>>(pt, pt2, b, glabel, (h-2)/(bn*2));
}
