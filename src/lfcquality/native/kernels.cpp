#include "kernels.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef LFQ_FFTW
#include <fftw3.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
thread_local std::string error_message;
constexpr int64_t block_size = 8192;
constexpr double undefined_value = std::numeric_limits<double>::quiet_NaN();
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
template<class F> int guarded(F fn) {
    try { fn(); error_message.clear(); return 0; }
    catch (const std::exception& e) { error_message = e.what(); return -1; }
    catch (...) { error_message = "unknown native analysis error"; return -1; }
}
void check_threads(int threads) {
    require(threads > 0, "threads must be positive");
#ifndef _OPENMP
    require(threads == 1, "this library was built without OpenMP");
#endif
}
int64_t elements(const int64_t* shape, int ndim) {
    require(ndim >= 1 && ndim <= 3, "only 1D/2D/3D fields are supported");
    int64_t n = 1;
    for (int d=0; d<ndim; ++d) {
        require(shape[d] > 0 && n <= INT64_MAX / shape[d], "invalid shape");
        n *= shape[d];
    }
    return n;
}
template<size_t K> using Row = std::array<double, K>;
// Fixed blocks and an ordered merge make reductions independent of OMP scheduling.
template<size_t K> Row<K> merge(const std::vector<Row<K>>& rows) {
    std::array<long double,K> sums{};
    for (const auto& row: rows) for (size_t j=0;j<K;++j) sums[j] += row[j];
    Row<K> result{};
    for (size_t j=0;j<K;++j) result[j] = static_cast<double>(sums[j]);
    return result;
}
template<class T> void statistics(const T* x, const T* y, int64_t n, int threads, double* o) {
    require(n > 0, "empty field");
    double lx=INFINITY, ly=INFINITY, le=INFINITY, hx=-INFINITY, hy=-INFINITY, he=-INFINITY;
    int invalid=0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:lx,ly,le) reduction(max:hx,hy,he) reduction(|:invalid)
    for (int64_t i=0;i<n;++i) {
        const double a=x[i], b=y[i], e=b-a;
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(e)) { invalid=1; continue; }
        lx=std::min(lx,a); hx=std::max(hx,a);
        ly=std::min(ly,b); hy=std::max(hy,b);
        le=std::min(le,e); he=std::max(he,e);
    }
    require(!invalid, "non-finite input or float64 error overflow");
    const double rx=hx-lx, ry=hy-ly, re=he-le;
    require(std::isfinite(rx) && std::isfinite(ry) && std::isfinite(re), "float64 range overflow");
    const double sx=rx>0?rx:1, sy=ry>0?ry:1, se=re>0?re:1;
    const double maxe=std::max(std::abs(le),std::abs(he));
    const int64_t nb=(n-1)/block_size+1;
    std::vector<Row<4>> partial(nb);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t k=0;k<nb;++k) {
        Row<4> s{};
        for (int64_t i=k*block_size;i<std::min(n,(k+1)*block_size);++i) {
            const double e=double(y[i])-double(x[i]);
            s[0]+=(double(x[i])-lx)/sx; s[1]+=(double(y[i])-ly)/sy;
            s[2]+=(e-le)/se; const double q=maxe>0?e/maxe:0; s[3]+=q*q;
        }
        partial[k]=s;
    }
    const auto s=merge(partial);
    const double mx=s[0]/n, my=s[1]/n, me=s[2]/n;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t k=0;k<nb;++k) {
        Row<4> v{};
        for (int64_t i=k*block_size;i<std::min(n,(k+1)*block_size);++i) {
            const double a=(double(x[i])-lx)/sx-mx, b=(double(y[i])-ly)/sy-my;
            const double e=(double(y[i])-double(x[i])-le)/se-me;
            v[0]+=a*a; v[1]+=b*b; v[2]+=e*e; v[3]+=a*b;
        }
        partial[k]=v;
    }
    const auto v=merge(partial);
    const double values[]={lx,ly,le,sx,sy,se,mx,my,me,v[0]/n,v[1]/n,v[2]/n,v[3]/n,
                           maxe*std::sqrt(s[3]/n),maxe,le+se*me,hx,hy};
    std::copy(values,values+18,o);
}
template<class T> void autocorrelation(const T* x, const T* y, const int64_t* sh, int ndim,
                                      int axis, int64_t lag, bool periodic, const double* s, int threads, double* o) {
    const int64_t n=elements(sh,ndim);
    require(axis>=0 && axis<ndim && lag>=0 && lag<sh[axis], "invalid ACF axis or lag");
    int64_t stride=1;
    for (int d=axis+1;d<ndim;++d) stride*=sh[d];
    const int64_t nb=(n-1)/block_size+1;
    std::vector<Row<3>> partial(nb);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t k=0;k<nb;++k) {
        Row<3> v{};
        for (int64_t i=k*block_size;i<std::min(n,(k+1)*block_size);++i) {
            const int64_t c=(i/stride)%sh[axis];
            if (!periodic && c+lag>=sh[axis]) continue;
            const int64_t j=i+(c+lag<sh[axis]?lag:lag-sh[axis])*stride;
            const double a=(double(x[i])-s[0])/s[3]-s[6], b=(double(x[j])-s[0])/s[3]-s[6];
            const double c0=(double(y[i])-s[1])/s[4]-s[7], d=(double(y[j])-s[1])/s[4]-s[7];
            const double e=(double(y[i])-double(x[i])-s[2])/s[5]-s[8];
            const double f=(double(y[j])-double(x[j])-s[2])/s[5]-s[8];
            v[0]+=a*b; v[1]+=c0*d; v[2]+=e*f;
        }
        partial[k]=v;
    }
    const auto v=merge(partial);
    for (int j=0;j<3;++j) o[j]=s[9+j]>0?v[j]/n/s[9+j]:undefined_value;
}
template<class T> void ssim(const T* x, const T* y, const int64_t* sh, int ndim, int window,
                           int stride, double range, const double* s, int threads, double* o) {
    elements(sh,ndim);
    require(window>0 && stride>0 && range>0 && std::isfinite(range), "invalid SSIM parameters");
    std::array<int64_t,3> dims{1,1,1}, grid{1,1,1}, win{1,1,1};
    for (int d=0;d<ndim;++d) {
        require(sh[d]>=window, "SSIM window exceeds a field dimension");
        dims[3-ndim+d]=sh[d]; grid[3-ndim+d]=(sh[d]-window)/stride+1; win[3-ndim+d]=window;
    }
    const int64_t nw=grid[0]*grid[1]*grid[2], count=win[0]*win[1]*win[2];
    const double scale=std::max({std::abs(s[0]),std::abs(s[1]),std::abs(s[16]),std::abs(s[17]),range});
    const double c1=std::pow(0.01*(range/scale),2), c2=std::pow(0.03*(range/scale),2);
    require(c1>0 && c2>0, "SSIM constants underflow at this dynamic range");
    const int64_t chunk=64, nb=(nw-1)/chunk+1;
    std::vector<Row<1>> partial(nb);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t k=0;k<nb;++k) {
        double sum=0;
        for (int64_t p=k*chunk;p<std::min(nw,(k+1)*chunk);++p) {
            const int64_t iz=(p/(grid[1]*grid[2]))*stride, iy=(p/grid[2]%grid[1])*stride, ix=(p%grid[2])*stride;
            const int64_t start=(iz*dims[1]+iy)*dims[2]+ix;
            const double ax=x[start], ay=y[start];
            double dx=0,dy=0;
            for (int64_t z=0;z<win[0];++z) for (int64_t yy=0;yy<win[1];++yy) for (int64_t xx=0;xx<win[2];++xx) {
                const int64_t i=start+(z*dims[1]+yy)*dims[2]+xx;
                dx+=(double(x[i])-ax)/scale; dy+=(double(y[i])-ay)/scale;
            }
            dx/=count; dy/=count;
            const double mx=ax/scale+dx, my=ay/scale+dy;
            double vx=0,vy=0,cov=0;
            for (int64_t z=0;z<win[0];++z) for (int64_t yy=0;yy<win[1];++yy) for (int64_t xx=0;xx<win[2];++xx) {
                const int64_t i=start+(z*dims[1]+yy)*dims[2]+xx;
                const double a=(double(x[i])-ax)/scale-dx, b=(double(y[i])-ay)/scale-dy;
                vx+=a*a; vy+=b*b; cov+=a*b;
            }
            const double lum=(2*mx*my+c1)/(mx*mx+my*my+c1);
            const double structural=(2*(cov/count)+c2)/(vx/count+vy/count+c2);
            sum+=std::clamp(lum*structural,-1.0,1.0);
        }
        partial[k][0]=sum;
    }
    o[0]=merge(partial)[0]/nw; o[1]=static_cast<double>(nw);
}
template<class X,class Y> void cross(const X* x,const Y* y,int64_t n,const double* a,const double* b,int threads,double* o) {
    // Each moment tuple is [min, scale, normalized_mean, normalized_variance].
    if (a[3]==0 || b[3]==0) { o[0]=undefined_value; return; }
    const int64_t nb=(n-1)/block_size+1;
    std::vector<Row<1>> partial(nb);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t k=0;k<nb;++k) {
        double v=0;
        for (int64_t i=k*block_size;i<std::min(n,(k+1)*block_size);++i)
            v+=((double(x[i])-a[0])/a[1]-a[2])*((double(y[i])-b[0])/b[1]-b[2]);
        partial[k][0]=v;
    }
    o[0]=std::clamp(merge(partial)[0]/n/std::sqrt(a[3])/std::sqrt(b[3]),-1.0,1.0);
}
template<class T> void prepare(const T* x,int64_t n,double lo,double scale,double mean,bool demean,int threads,double* o) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i=0;i<n;++i) o[i]=demean?(double(x[i])-lo)/scale-mean:double(x[i])/scale;
}
}

extern "C" {
int lfq_abi_version() { return 1; }
int lfq_openmp() {
#ifdef _OPENMP
    return _OPENMP;
#else
    return 0;
#endif
}
int lfq_fftw() {
#ifdef LFQ_FFTW
    return 1;
#else
    return 0;
#endif
}
int lfq_fftw_threads() {
#ifdef LFQ_FFTW_THREADS
    return 1;
#else
    return 0;
#endif
}
const char* lfq_fftw_version() {
#ifdef LFQ_FFTW
    return fftw_version;
#else
    return "unavailable";
#endif
}
const char* lfq_last_error() { return error_message.c_str(); }
int lfq_team_size(int requested) {
    int result=1;
    if (requested<=0) return 0;
    #pragma omp parallel num_threads(requested)
    {
        #ifdef _OPENMP
        #pragma omp single
        result=omp_get_num_threads();
        #endif
    }
    return result;
}
int lfq_stats(const void* x,const void* y,int bytes,int64_t n,int threads,double* o) {
    return guarded([&]{check_threads(threads); require(bytes==4||bytes==8,"unsupported dtype");
        if(bytes==4) statistics((const float*)x,(const float*)y,n,threads,o);
        else statistics((const double*)x,(const double*)y,n,threads,o);});
}
int lfq_basic(const double* s,double range,double* o) {
    return guarded([&]{
        require(range>=0 && std::isfinite(range),"invalid data range");
        o[0]=range>0?s[13]/range:undefined_value;
        o[1]=range>0?(s[13]>0?20*(std::log10(range)-std::log10(s[13])):INFINITY):undefined_value;
        o[2]=s[9]>0 && s[10]>0?std::clamp(s[12]/std::sqrt(s[9])/std::sqrt(s[10]),-1.0,1.0):undefined_value;
        o[3]=1-o[2];
    });
}
int lfq_acf(const void* x,const void* y,int bytes,const int64_t* sh,int ndim,int axis,int64_t lag,int periodic,const double* s,int threads,double* o) {
    return guarded([&]{check_threads(threads); require(bytes==4||bytes==8,"unsupported dtype");
        if(bytes==4) autocorrelation((const float*)x,(const float*)y,sh,ndim,axis,lag,periodic,s,threads,o);
        else autocorrelation((const double*)x,(const double*)y,sh,ndim,axis,lag,periodic,s,threads,o);});
}
int lfq_ssim(const void* x,const void* y,int bytes,const int64_t* sh,int ndim,int window,int stride,double range,const double* s,int threads,double* o) {
    return guarded([&]{check_threads(threads); require(bytes==4||bytes==8,"unsupported dtype");
        if(bytes==4) ssim((const float*)x,(const float*)y,sh,ndim,window,stride,range,s,threads,o);
        else ssim((const double*)x,(const double*)y,sh,ndim,window,stride,range,s,threads,o);});
}
int lfq_cross(const void* x,int xb,const void* y,int yb,int64_t n,const double* a,const double* b,int threads,double* o) {
    return guarded([&]{check_threads(threads); require(n>0 && (xb==4||xb==8) && (yb==4||yb==8),"invalid cross-correlation input");
        if(xb==4 && yb==4) cross((const float*)x,(const float*)y,n,a,b,threads,o);
        else if(xb==4) cross((const float*)x,(const double*)y,n,a,b,threads,o);
        else if(yb==4) cross((const double*)x,(const float*)y,n,a,b,threads,o);
        else cross((const double*)x,(const double*)y,n,a,b,threads,o);});
}
int lfq_prepare_fft(const void* x,int bytes,int64_t n,double lo,double scale,double mean,int demean,int threads,double* o) {
    return guarded([&]{check_threads(threads); require(n>0 && scale>0 && std::isfinite(scale),"invalid FFT scaling"); require(bytes==4||bytes==8,"unsupported dtype");
        if(bytes==4) prepare((const float*)x,n,lo,scale,mean,demean,threads,o);
        else prepare((const double*)x,n,lo,scale,mean,demean,threads,o);});
}
int lfq_spectrum(const double* x,const double* y,const int64_t* sh,int ndim,const double* spacing,int bins,int threads,double* totals,double* shells) {
    return guarded([&]{
        check_threads(threads); const int64_t n=elements(sh,ndim); require(bins>0,"invalid spectral bins");
        const int64_t last=sh[ndim-1]/2+1, m=n/sh[ndim-1]*last;
        double rmax=0;
        for(int d=0;d<ndim;++d) {require(spacing[d]>0 && std::isfinite(spacing[d]),"invalid spacing");
            const double f=double(sh[d]/2)/sh[d]/spacing[d]; rmax=std::hypot(rmax,f);}
        require(std::isfinite(rmax) && (rmax>0 || n==1),"frequency range overflow or underflow");
        const int64_t chunk=65536, nb=(m-1)/chunk+1, width=6+3*int64_t(bins);
        std::vector<double> partial(nb*width,0);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int64_t k=0;k<nb;++k) {
            double* p=partial.data()+k*width;
            for(int64_t i=k*chunk;i<std::min(m,(k+1)*chunk);++i) {
                int64_t q=i;
                double radius=0;
                for(int d=ndim-1;d>=0;--d) {
                    const int64_t len=d==ndim-1?last:sh[d], c=q%len; q/=len;
                    const int64_t freq=(d==ndim-1 || c<=sh[d]/2)?c:c-sh[d];
                    const double f=double(freq)/sh[d]/spacing[d]; radius=std::hypot(radius,f);
                }
                const int64_t z=i%last;
                const double weight=z==0 || (sh[ndim-1]%2==0 && z==sh[ndim-1]/2)?1:2;
                const int bin=rmax>0?std::min(bins-1,int(radius/rmax*bins)):0;
                const double px=x[2*i]*x[2*i]+x[2*i+1]*x[2*i+1], py=y[2*i]*y[2*i]+y[2*i+1]*y[2*i+1];
                const double diff=py-px, dr=y[2*i]-x[2*i], di=y[2*i+1]-x[2*i+1];
                p[0]+=weight*px; p[1]+=weight*py; p[2]+=weight*std::abs(diff);
                p[3]+=weight*diff*diff; p[4]+=weight*px*px; p[5]+=weight*(dr*dr+di*di);
                p[6+bin]+=weight*px; p[6+bins+bin]+=weight*py; p[6+2*bins+bin]+=weight;
            }
        }
        for(int64_t j=0;j<width;++j) {
            long double v=0; for(int64_t k=0;k<nb;++k) v+=partial[k*width+j];
            if(j<6) totals[j]=double(v); else shells[j-6]=double(v);
        }
    });
}
int lfq_spectral_fields(const void* x,const void* y,int bytes,const int64_t* sh,int ndim,
                        const double* s,const double* spacing,int bins,int demean,
                        int threads,int fft_threads,double* totals,double* shells) {
    return guarded([&]{
        check_threads(threads);
#ifdef LFQ_FFTW
        require(fft_threads>0,"FFT threads must be positive");
#ifndef LFQ_FFTW_THREADS
        require(fft_threads==1,"FFTW thread library is unavailable");
#endif
        const int64_t n=elements(sh,ndim), m=n/sh[ndim-1]*(sh[ndim-1]/2+1);
        require(uint64_t(n)<=SIZE_MAX/sizeof(double) && uint64_t(m)<=SIZE_MAX/(2*sizeof(double)),"FFT allocation overflow");
        int dims[3]; for(int d=0;d<ndim;++d) {require(sh[d]<=INT32_MAX,"FFTW dimension exceeds INT_MAX"); dims[d]=int(sh[d]);}
        const double scale=std::max({std::abs(s[0]),std::abs(s[1]),std::abs(s[16]),std::abs(s[17])});
        const double factor=scale>0?scale:1;
        using Buffer=std::unique_ptr<double,decltype(&fftw_free)>;
        Buffer work(fftw_alloc_real(n),fftw_free), fx(fftw_alloc_real(2*m),fftw_free), fy(fftw_alloc_real(2*m),fftw_free);
        if(!work || !fx || !fy) throw std::bad_alloc();
        // FFTW planning and destroying plans are serialized; execution is reentrant.
        // We never change process-wide OMP settings or destroy FFTW's global state.
        static std::mutex planner_mutex;
        auto transform=[&](const void* data,int field,double* dest) {
            if(lfq_prepare_fft(data,bytes,n,s[field],factor,s[6+field]*(s[3+field]/factor),demean,threads,work.get()))
                throw std::runtime_error(error_message);
            fftw_plan plan;
            {
                std::lock_guard<std::mutex> lock(planner_mutex);
#ifdef LFQ_FFTW_THREADS
                static const int initialized=fftw_init_threads();
                require(initialized!=0,"FFTW thread initialization failed");
                fftw_plan_with_nthreads(fft_threads);
#endif
                plan=fftw_plan_dft_r2c(ndim,dims,work.get(),reinterpret_cast<fftw_complex*>(dest),FFTW_ESTIMATE);
            }
            require(plan!=nullptr,"FFTW plan creation failed");
            fftw_execute(plan);
            {std::lock_guard<std::mutex> lock(planner_mutex); fftw_destroy_plan(plan);}
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(int64_t i=0;i<2*m;++i) dest[i]/=n;
        };
        transform(x,0,fx.get()); transform(y,1,fy.get());
        if(lfq_spectrum(fx.get(),fy.get(),sh,ndim,spacing,bins,threads,totals,shells)) throw std::runtime_error(error_message);
        totals[6]=factor;
        totals[7]=totals[0]>0?totals[2]/totals[0]:undefined_value;
        totals[8]=totals[4]>0?std::sqrt(totals[3]/totals[4]):undefined_value;
        totals[9]=totals[0]>0?std::sqrt(totals[5]/totals[0]):undefined_value;
#else
        throw std::runtime_error("built without FFTW; rebuild with --fftw required");
#endif
    });
}
}
