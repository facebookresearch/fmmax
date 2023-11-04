"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5267],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>b});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},u="mdxType",f={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,c=e.parentName,p=o(e,["components","mdxType","originalType","parentName"]),u=l(n),m=i,b=u["".concat(c,".").concat(m)]||u[m]||f[m]||a;return n?r.createElement(b,s(s({ref:t},p),{},{components:n})):r.createElement(b,s({ref:t},p))}));function b(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,s=new Array(a);s[0]=m;var o={};for(var c in t)hasOwnProperty.call(t,c)&&(o[c]=t[c]);o.originalType=e,o[u]="string"==typeof e?e:i,s[1]=o;for(var l=2;l<a;l++)s[l]=n[l];return r.createElement.apply(null,s)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},86:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>s,default:()=>f,frontMatter:()=>a,metadata:()=>o,toc:()=>l});var r=n(7462),i=(n(7294),n(3905));const a={id:"basis.Expansion"},s=void 0,o={unversionedId:"API/basis.Expansion",id:"API/basis.Expansion",title:"basis.Expansion",description:"Class basis.Expansion():",source:"@site/docs/API/basis.Expansion.md",sourceDirName:"API",slug:"/API/basis.Expansion",permalink:"/fmmax/API/basis.Expansion",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/basis.Expansion.md",tags:[],version:"current",frontMatter:{id:"basis.Expansion"},sidebar:"APISidebar",previous:{title:"basis.LatticeVectors.reciprocal",permalink:"/fmmax/API/basis.LatticeVectors.reciprocal"},next:{title:"basis.LatticeVectors",permalink:"/fmmax/API/basis.LatticeVectors"}},c={},l=[{value:"<code>Class basis.Expansion():</code>",id:"class-basisexpansion",level:3},{value:"Args:",id:"args",level:4}],p={toc:l},u="wrapper";function f(e){let{components:t,...n}=e;return(0,i.kt)(u,(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h3",{id:"class-basisexpansion"},(0,i.kt)("inlineCode",{parentName:"h3"},"Class basis.Expansion():")),(0,i.kt)("p",null,"Stores the expansion."),(0,i.kt)("p",null,"The expansion consists of the integer coefficients of the reciprocal lattice vectors\nused in the Fourier expansion of fields, permittivities, etc. in the FMM scheme."),(0,i.kt)("h4",{id:"args"},"Args:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"basis_coefficients"),": The integer coefficients of the primitive reciprocal lattice\nvectors, which generate the full set of reciprocal-space vectors in the expansion."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"num_terms"),": The number of terms in the expansion.")))}f.isMDXComponent=!0}}]);