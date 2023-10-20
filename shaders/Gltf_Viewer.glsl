precision highp float;
precision highp int;
precision highp sampler2D;

uniform float uCameraUnderWater;
uniform float uWaterLevel;
uniform vec3 uSunDirection;
uniform sampler2D t_PerlinNoise;

uniform sampler2D tTriangleTexture;
uniform sampler2D tAABBTexture;
uniform sampler2D tAlbedoTextures[8]; // 8 = max number of diffuse albedo textures per model
uniform sampler2D tHDRTexture;
uniform float uSkyLightIntensity;
uniform float uSunLightIntensity;
uniform vec3 uSunColor;
uniform bool uSceneIsDynamic;

// (1 / 2048 texture width)
#define INV_TEXTURE_WIDTH 0.00048828125

vec3 rayOrigin, rayDirection;
// recorded intersection data:
vec3 hitNormal, hitEmission, hitColor;
vec2 hitUV;
float hitObjectID, hitOpacity;
int hitType = -100; 
int hitAlbedoTextureID;

struct Quad { vec3 v0; vec3 v1; vec3 v2; vec3 v3; vec3 emission; vec3 color; int type; };
struct Box { vec3 minCorner; vec3 maxCorner; vec3 emission; vec3 color; int type; };
Box box;


#include <pathtracing_uniforms_and_defines>

#include <pathtracing_skymodel_defines>

#include <pathtracing_plane_intersect>

#include <pathtracing_physical_sky_functions>

#include <pathtracing_random_functions>

#include <pathtracing_calc_fresnel_reflectance>

#include <pathtracing_sphere_intersect>

#include <pathtracing_box_intersect>

#include <pathtracing_boundingbox_intersect>

#include <pathtracing_bvhTriangle_intersect>


vec2 stackLevels[28];

//vec4 boxNodeData0 corresponds to .x = idTriangle,  .y = aabbMin.x, .z = aabbMin.y, .w = aabbMin.z
//vec4 boxNodeData1 corresponds to .x = idRightChild .y = aabbMax.x, .z = aabbMax.y, .w = aabbMax.z

void GetBoxNodeData(const in float i, inout vec4 boxNodeData0, inout vec4 boxNodeData1)
{
	// each bounding box's data is encoded in 2 rgba(or xyzw) texture slots 
	float ix2 = i * 2.0;
	// (ix2 + 0.0) corresponds to .x = idTriangle,  .y = aabbMin.x, .z = aabbMin.y, .w = aabbMin.z 
	// (ix2 + 1.0) corresponds to .x = idRightChild .y = aabbMax.x, .z = aabbMax.y, .w = aabbMax.z 

	ivec2 uv0 = ivec2( mod(ix2 + 0.0, 2048.0), (ix2 + 0.0) * INV_TEXTURE_WIDTH ); // data0
	ivec2 uv1 = ivec2( mod(ix2 + 1.0, 2048.0), (ix2 + 1.0) * INV_TEXTURE_WIDTH ); // data1
	
	boxNodeData0 = texelFetch(tAABBTexture, uv0, 0);
	boxNodeData1 = texelFetch(tAABBTexture, uv1, 0);
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
float SceneIntersectViewer( )
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
	vec4 currentBoxNodeData0, nodeAData0, nodeBData0, tmpNodeData0;
	vec4 currentBoxNodeData1, nodeAData1, nodeBData1, tmpNodeData1;
	
	vec4 vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;

	vec3 inverseDir = 1.0 / rayDirection;
	vec3 normal, n;

	vec2 currentStackData, stackDataA, stackDataB, tmpStackData;
	ivec2 uv0, uv1, uv2, uv3, uv4, uv5, uv6, uv7;

	float d;
	float t = INFINITY;
        float stackptr = 0.0;
	float id = 0.0;
	float tu, tv;
	float triangleID = 0.0;
	float triangleU = 0.0;
	float triangleV = 0.0;
	float triangleW = 0.0;

	int objectCount = 0;
	
	hitObjectID = -INFINITY;

	int skip = FALSE;
	int triangleLookupNeeded = FALSE;
	int isRayExiting = FALSE;


	// GROUND Plane (thin, wide box that acts like ground plane)
	// d = BoxIntersect( box.minCorner, box.maxCorner, rayOrigin, rayDirection, n, isRayExiting );
	// if (d < t)
	// {
	// 	t = d;
	// 	hitNormal = n;
	// 	hitEmission = box.emission;
	// 	hitColor = box.color;
	// 	hitType = box.type;
	// 	hitObjectID = float(objectCount);
	// }
	// objectCount++;
	

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// glTF
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	GetBoxNodeData(stackptr, currentBoxNodeData0, currentBoxNodeData1);
	currentStackData = vec2(stackptr, BoundingBoxIntersect(currentBoxNodeData0.yzw, currentBoxNodeData1.yzw, rayOrigin, inverseDir));
	stackLevels[0] = currentStackData;
	skip = (currentStackData.y < t) ? TRUE : FALSE;

	while (true)
        {
		if (skip == FALSE) 
                {
                        // decrease pointer by 1 (0.0 is root level, 27.0 is maximum depth)
                        if (--stackptr < 0.0) // went past the root level, terminate loop
                                break;

                        currentStackData = stackLevels[int(stackptr)];
			
			if (currentStackData.y >= t)
				continue;
			
			GetBoxNodeData(currentStackData.x, currentBoxNodeData0, currentBoxNodeData1);
                }
		skip = FALSE; // reset skip
		
		if (currentBoxNodeData0.x < 0.0) // // < 0.0 signifies an inner node 
		{
			GetBoxNodeData(currentStackData.x + 1.0, nodeAData0, nodeAData1);
			GetBoxNodeData(currentBoxNodeData1.x, nodeBData0, nodeBData1);
			stackDataA = vec2(currentStackData.x + 1.0, BoundingBoxIntersect(nodeAData0.yzw, nodeAData1.yzw, rayOrigin, inverseDir));
			stackDataB = vec2(currentBoxNodeData1.x, BoundingBoxIntersect(nodeBData0.yzw, nodeBData1.yzw, rayOrigin, inverseDir));
			
			// first sort the branch node data so that 'a' is the smallest
			if (stackDataB.y < stackDataA.y)
			{
				tmpStackData = stackDataB;
				stackDataB = stackDataA;
				stackDataA = tmpStackData;

				tmpNodeData0 = nodeBData0;   tmpNodeData1 = nodeBData1;
				nodeBData0   = nodeAData0;   nodeBData1   = nodeAData1;
				nodeAData0   = tmpNodeData0; nodeAData1   = tmpNodeData1;
			} // branch 'b' now has the larger rayT value of 'a' and 'b'

			if (stackDataB.y < t) // see if branch 'b' (the larger rayT) needs to be processed
			{
				currentStackData = stackDataB;
				currentBoxNodeData0 = nodeBData0;
				currentBoxNodeData1 = nodeBData1;
				skip = TRUE; // this will prevent the stackptr from decreasing by 1
			}
			if (stackDataA.y < t) // see if branch 'a' (the smaller rayT) needs to be processed 
			{
				if (skip == TRUE) // if larger branch 'b' needed to be processed also,
					stackLevels[int(stackptr++)] = stackDataB; // cue larger branch 'b' for future round
							// also, increase pointer by 1
				
				currentStackData = stackDataA;
				currentBoxNodeData0 = nodeAData0; 
				currentBoxNodeData1 = nodeAData1;
				skip = TRUE; // this will prevent the stackptr from decreasing by 1
			}

			continue;
		} // end if (currentBoxNodeData0.x < 0.0) // inner node


		// else this is a leaf

		// each triangle's data is encoded in 8 rgba(or xyzw) texture slots
		
		id = 8.0 * currentBoxNodeData0.x;

		uv0 = ivec2( mod(id + 0.0, 2048.0), (id + 0.0) * INV_TEXTURE_WIDTH );
		uv1 = ivec2( mod(id + 1.0, 2048.0), (id + 1.0) * INV_TEXTURE_WIDTH );
		uv2 = ivec2( mod(id + 2.0, 2048.0), (id + 2.0) * INV_TEXTURE_WIDTH );
		
		vd0 = texelFetch(tTriangleTexture, uv0, 0);
		vd1 = texelFetch(tTriangleTexture, uv1, 0);
		vd2 = texelFetch(tTriangleTexture, uv2, 0);

		d = BVH_TriangleIntersect( vec3(vd0.xyz), vec3(vd0.w, vd1.xy), vec3(vd1.zw, vd2.x), rayOrigin, rayDirection, tu, tv );

		if (d < t)
		{
			t = d;
			triangleID = id;
			triangleU = tu;
			triangleV = tv;
			triangleLookupNeeded = TRUE;
		}
	      
        } // end while (TRUE)


	if (triangleLookupNeeded == TRUE)
	{
		uv0 = ivec2( mod(triangleID + 0.0, 2048.0), floor((triangleID + 0.0) * INV_TEXTURE_WIDTH) );
		uv1 = ivec2( mod(triangleID + 1.0, 2048.0), floor((triangleID + 1.0) * INV_TEXTURE_WIDTH) );
		uv2 = ivec2( mod(triangleID + 2.0, 2048.0), floor((triangleID + 2.0) * INV_TEXTURE_WIDTH) );
		uv3 = ivec2( mod(triangleID + 3.0, 2048.0), floor((triangleID + 3.0) * INV_TEXTURE_WIDTH) );
		uv4 = ivec2( mod(triangleID + 4.0, 2048.0), floor((triangleID + 4.0) * INV_TEXTURE_WIDTH) );
		uv5 = ivec2( mod(triangleID + 5.0, 2048.0), floor((triangleID + 5.0) * INV_TEXTURE_WIDTH) );
		uv6 = ivec2( mod(triangleID + 6.0, 2048.0), floor((triangleID + 6.0) * INV_TEXTURE_WIDTH) );
		uv7 = ivec2( mod(triangleID + 7.0, 2048.0), floor((triangleID + 7.0) * INV_TEXTURE_WIDTH) );
		
		vd0 = texelFetch(tTriangleTexture, uv0, 0);
		vd1 = texelFetch(tTriangleTexture, uv1, 0);
		vd2 = texelFetch(tTriangleTexture, uv2, 0);
		vd3 = texelFetch(tTriangleTexture, uv3, 0);
		vd4 = texelFetch(tTriangleTexture, uv4, 0);
		vd5 = texelFetch(tTriangleTexture, uv5, 0);
		vd6 = texelFetch(tTriangleTexture, uv6, 0);
		vd7 = texelFetch(tTriangleTexture, uv7, 0);

		// face normal for flat-shaded polygon look
		// hitNormal = ( cross(vec3(vd0.w, vd1.xy) - vec3(vd0.xyz), vec3(vd1.zw, vd2.x) - vec3(vd0.xyz)) );

		// interpolated normal using triangle intersection's uv's
		triangleW = 1.0 - triangleU - triangleV;
		hitNormal = (triangleW * vec3(vd2.yzw) + triangleU * vec3(vd3.xyz) + triangleV * vec3(vd3.w, vd4.xy));
		// if (hitNormal.y > 0.1){
		// hitNormal = vec3(1.,0.,0.);
		// }
		hitEmission = vec3(1, 0, 1); // use this if hitType will be LIGHT
		hitColor = vd6.yzw;
		hitOpacity = vd7.y;
		hitUV = triangleW * vec2(vd4.zw) + triangleU * vec2(vd5.xy) + triangleV * vec2(vd5.zw);
		hitType = int(vd6.x);
		hitAlbedoTextureID = int(vd7.x);
		hitObjectID = float(objectCount);
	}

	return t;

} 


vec3 Get_HDR_Color(vec3 rayDirection)
{
	vec2 sampleUV;
	sampleUV.x = atan(rayDirection.z, rayDirection.x) * ONE_OVER_TWO_PI + 0.5;
	sampleUV.y = asin(clamp(rayDirection.y, -1.0, 1.0)) * ONE_OVER_PI + 0.5;
	
	return texture( tHDRTexture, sampleUV ).rgb;
}

/*
//-----------------------------------------------------------------------------------------------------------------------------
vec3 CalculateRadianceViewer( out vec3 objectNormal, out vec3 objectColor, out float objectID, out float pixelSharpness )
//-----------------------------------------------------------------------------------------------------------------------------
{
	vec3 randVec = vec3(rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0);

	vec3 accumCol = vec3(0.0);
	vec3 mask = vec3(1.0);
	vec3 reflectionMask = vec3(1);
	vec3 reflectionRayOrigin = vec3(0);
	vec3 reflectionRayDirection = vec3(0);
	vec3 n, nl, x;
	vec3 firstX = vec3(0);
	vec3 tdir;

	float hitDistance;
	float nc, nt, ratioIoR, Re, Tr;
	//float P, RP, TP;
	float weight;
	float t = INFINITY;
	float epsIntersect = 0.01;
	
	int diffuseCount = 0;
	int previousIntersecType = -100;
	hitType = -100;
	
	int bounceIsSpecular = TRUE;
	int sampleLight = FALSE;
	int willNeedReflectionRay = FALSE;

    	for (int bounces = 0; bounces < 5; bounces++)
	{
		previousIntersecType = hitType;

		t = SceneIntersectViewer();

		if (t == INFINITY)
		{
			// ray hits sky first
			if (bounces == 0)
			{
				pixelSharpness = 1.01;

				accumCol += Get_HDR_Color(rayDirection);
				break;
			}

			// if ray bounced off of diffuse material and hits sky
			if (previousIntersecType == DIFF)
			{
				if (sampleLight == TRUE)
					accumCol += mask * uSunColor * uSunLightIntensity * 0.5;
				else
					accumCol += mask * Get_HDR_Color(rayDirection) * uSkyLightIntensity * 0.5;
			}

			// if ray bounced off of glass and hits sky
			if (previousIntersecType == REFR)
			{
				if (diffuseCount == 0) // camera looking through glass, hitting the sky
				{
					pixelSharpness = 1.01;
					mask *= Get_HDR_Color(rayDirection);
				}	
				else if (sampleLight == TRUE) // sun rays going through glass, hitting another surface
					mask *= uSunColor * uSunLightIntensity;
				else  // sky rays going through glass, hitting another surface
					mask *= Get_HDR_Color(rayDirection) * uSkyLightIntensity;

				if (bounceIsSpecular == TRUE) // prevents sun 'fireflies' on diffuse surfaces
					accumCol += mask;
			}

			if (willNeedReflectionRay == TRUE)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				diffuseCount = 0;
				continue;
			}
			// reached a light, so we can exit
			break;

		} // end if (t == INFINITY)


		 // other lights, like houselights, could be added to the scene
		// if we reached light material, don't spawn any more rays
		// if (hitType == LIGHT)
		// {
	    // 		accumCol = mask * hitEmission * 0.5;
		// 	break;
		// } 

		// Since we want fast direct sunlight caustics through windows, we don't use the following early out
		// if (sampleLight == TRUE)
		// 	break; 

		// useful data
		n = normalize(hitNormal);
		nl = dot(n, rayDirection) < 0.0 ? n : -n;
		x = rayOrigin + rayDirection * t;

		if (bounces == 0)
		{
			objectNormal = nl;
			objectColor = hitColor;
			objectID = hitObjectID;
		}
		// if (bounces == 1 && diffuseCount == 0)
		// {
		// 	objectNormal = nl;
		// }



		if (hitType == DIFF) // Ideal DIFFUSE reflection
		{
			// if (diffuseCount == 0)
			// 	objectColor = hitColor;

			diffuseCount++;

			mask *= hitColor;
	    		bounceIsSpecular = FALSE;

			if (diffuseCount == 1 && rand() < 0.5)
			{
				mask *= 2.0;
				// this branch gathers color bleeding / caustics from other surfaces hit in the future
				// choose random Diffuse sample vector
				rayDirection = randomCosWeightedDirectionInHemisphere(nl);
				rayOrigin = x + nl * epsIntersect;

				continue;
			}
			
			// this branch acts like a traditional shadowRay, checking for direct light from the Sun..
			// if it has a clear path and hits the Sun on the next bounce, sunlight is gathered, otherwise returns black (shadow)
			rayDirection = normalize(uSunDirection + (randVec * 0.01));
			rayOrigin = x + nl * epsIntersect;

			weight = max(0.0, dot(rayDirection, nl));
			mask *= diffuseCount == 1 ? 2.0 : 1.0;
			mask *= weight;
			
			sampleLight = TRUE;
			continue;
			
		} // end if (hitType == DIFF)

		if (hitType == SPEC)  // Ideal SPECULAR reflection
		{
			mask *= hitColor;

			rayDirection = reflect(rayDirection, nl);
			rayOrigin = x + rayDirection * epsIntersect;

			//bounceIsSpecular = TRUE; // turn on mirror caustics
			continue;
		}

		if (hitType == REFR)  // Ideal dielectric REFRACTION
		{
			pixelSharpness = diffuseCount == 0 ? -1.0 : pixelSharpness;

			nc = 1.0; // IOR of Air
			nt = 1.5; // IOR of common Glass
			Re = calcFresnelReflectance(rayDirection, n, nc, nt, ratioIoR);
			Tr = 1.0 - Re;

			if (bounces == 0)// || (bounces == 1 && hitObjectID != objectID && bounceIsSpecular == TRUE))
			{
				reflectionMask = mask * Re;
				reflectionRayDirection = reflect(rayDirection, nl); // reflect ray from surface
				reflectionRayOrigin = x + nl * epsIntersect;
				willNeedReflectionRay = TRUE;
			}

			if (Re == 1.0)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				continue;
			}
			
			// transmit ray through surface
			
			//mask *= 1.0 - (hitColor * hitOpacity);
			mask *= hitColor;
			mask *= Tr;
			//tdir = refract(rayDirection, nl, ratioIoR);
			rayDirection = rayDirection; // TODO using rayDirection instead of tdir, because going through common Glass makes everything spherical from up close...
			rayOrigin = x + rayDirection * epsIntersect;

			if (diffuseCount < 2)
				bounceIsSpecular = TRUE;
			continue;
			

		} // end if (hitType == REFR)

	} // end for (int bounces = 0; bounces < 4; bounces++)

	return accumCol;
} // end vec3 CalculateRadianceViewer()
*/

//-----------------------------------------------------------------------
void SetupScene(void)
//-----------------------------------------------------------------------
{
	// Add thin box for the ground (acts like ground plane)
	box = Box( vec3(-400, -1, -400), vec3(400, 0, 400), vec3(0), vec3(0.15), DIFF);
}



//---------------------------------------------------------------------------------------------------------
float DisplacementBoxIntersect( vec3 minCorner, vec3 maxCorner, vec3 rayOrigin, vec3 rayDirection )
//---------------------------------------------------------------------------------------------------------
{
	vec3 invDir = 1.0 / rayDirection;
	vec3 tmin = (minCorner - rayOrigin) * invDir;
	vec3 tmax = (maxCorner - rayOrigin) * invDir;
	
	vec3 real_min = min(tmin, tmax);
	vec3 real_max = max(tmin, tmax);
	
	float minmax = min( min(real_max.x, real_max.y), real_max.z);
	float maxmin = max( max(real_min.x, real_min.y), real_min.z);
	
	// early out
	if (minmax < maxmin) return INFINITY;
	
	if (maxmin > 0.0) // if we are outside the box
	{
		return maxmin;	
	}
		
	if (minmax > 0.0) // else if we are inside the box
	{
		return minmax;
	}
				
	return INFINITY;
}

// WATER
/* Credit: some of the following water code is borrowed from https://www.shadertoy.com/view/Ms2SD1 posted by user 'TDM' */

#define WATER_COLOR vec3(0.96, 1.0, 0.98)
#define WATER_SAMPLE_SCALE 0.01 
#define WATER_WAVE_HEIGHT 4.0 // max height of water waves   
#define WATER_FREQ        0.1 // wave density: lower = spread out, higher = close together
#define WATER_CHOPPY      2.0 // smaller beachfront-type waves, they travel in parallel
#define WATER_SPEED       0.1 // how quickly time passes
#define OCTAVE_M  mat2(1.6, 1.2, -1.2, 1.6);
#define WATER_DETAIL_FAR 1000.0

// float noise( in vec2 p )
// {
// 	return texture(t_PerlinNoise, p).x;
// }

float sea_octave( vec2 uv, float choppy )
{
	uv += texture(t_PerlinNoise, uv).x * 2.0 - 1.0;        
	vec2 wv = 1.0 - abs(sin(uv));
	vec2 swv = abs(cos(uv));    
	wv = mix(wv, swv, wv);
	return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

float getOceanWaterHeight( vec3 p )
{
	float freq = WATER_FREQ;
	float amp = 1.0;
	float choppy = WATER_CHOPPY;
	float sea_time = uTime * WATER_SPEED;
	
	vec2 uv = p.xz * WATER_SAMPLE_SCALE; 
	uv.x *= 0.75;
	float d, h = 0.0;
	d =  sea_octave((uv + sea_time) * freq, choppy);
	d += sea_octave((uv - sea_time) * freq, choppy);
	h += d * amp;        
	
	return h * WATER_WAVE_HEIGHT + uWaterLevel;
}

float getOceanWaterHeight_Detail( vec3 p )
{
	float freq = WATER_FREQ;
	float amp = 1.0;
	float choppy = WATER_CHOPPY;
	float sea_time = uTime * WATER_SPEED;
	
	vec2 uv = p.xz * WATER_SAMPLE_SCALE; 
	uv.x *= 0.75;
	float d, h = 0.0;    
	for(int i = 0; i < 4; i++)
	{        
		d =  sea_octave((uv + sea_time) * freq, choppy);
		d += sea_octave((uv - sea_time) * freq, choppy);
		h += d * amp;        
		uv *= OCTAVE_M; freq *= 1.9; amp *= 0.22;
		choppy = mix(choppy, 1.0, 0.2);
	}
	return h * WATER_WAVE_HEIGHT + uWaterLevel;
}

// CLOUDS
/* Credit: some of the following cloud code is borrowed from https://www.shadertoy.com/view/XtBXDw posted by user 'valentingalea' */
#define THICKNESS      25.0
#define ABSORPTION     0.45
#define N_MARCH_STEPS  12
#define N_LIGHT_STEPS  3

const mat3 m = 1.21 * mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
		    -0.60, -0.48,  0.64 );

float fbm( vec3 p )
{
	float t;
	float mult = 2.0;
	t  = 1.0 * texture(t_PerlinNoise, p.xz).x;   p = m * p * mult;
	t += 0.5 * texture(t_PerlinNoise, p.xz).x;   p = m * p * mult;
	t += 0.25 * texture(t_PerlinNoise, p.xz).x;
	
	return t;
}

float cloud_density( vec3 pos, float cov )
{
	float dens = fbm(pos * 0.002);
	dens *= smoothstep(cov, cov + 0.05, dens);
	return clamp(dens, 0.0, 1.0);	
}

float cloud_light( vec3 pos, vec3 dir_step, float cov )
{
	float T = 1.0; // transmitance
    	float dens;
    	float T_i;
	
	for (int i = 0; i < N_LIGHT_STEPS; i++) 
	{
		dens = cloud_density(pos, cov);
		T_i = exp(-ABSORPTION * dens);
		T *= T_i;
		pos += dir_step;
	}
	return T;
}

vec4 render_clouds(vec3 rayOrigin, vec3 rayDirection)
{
	float march_step = THICKNESS / float(N_MARCH_STEPS);
	vec3 pos = rayOrigin + vec3(uTime * -3.0, uTime * -0.5, uTime * -2.0);
	vec3 dir_step = rayDirection / clamp(rayDirection.y, 0.3, 1.0) * march_step;
	vec3 light_step = uSunDirection * 5.0;
	
	float covAmount = (sin(mod(uTime * 0.1 + 3.5, TWO_PI))) * 0.5 + 0.5;
	float coverage = mix(1.1, 1.5, clamp(covAmount, 0.0, 1.0));
	float T = 1.0; // transmitance
	vec3 C = vec3(0); // color
	float alpha = 0.0;
	float dens;
	float T_i;
	float cloudLight;
	
	for (int i = 0; i < N_MARCH_STEPS; i++)
	{
		dens = cloud_density(pos, coverage);
		T_i = exp(-ABSORPTION * dens * march_step);
		T *= T_i;
		cloudLight = cloud_light(pos, light_step, coverage);
		C += T * cloudLight * dens * march_step;
		C = mix(C * 0.95, C, cloudLight);
		alpha += (1.0 - T_i) * (1.0 - alpha);
		pos += dir_step;
	}
	
	return vec4(C, alpha);
}

// TERRAIN
#define TERRAIN_HEIGHT 2000.0
#define TERRAIN_SAMPLE_SCALE 0.00004 
#define TERRAIN_LIFT -1300.0 // how much to lift or drop the entire terrain
#define TERRAIN_DETAIL_FAR 40000.0

float lookup_Heightmap( in vec3 pos )
{
	vec2 uv = pos.xz;
	uv *= TERRAIN_SAMPLE_SCALE;
	float h = 0.0;
	float mult = 1.0;
	for (int i = 0; i < 4; i ++)
	{
		h += mult * texture(t_PerlinNoise, uv + 0.5).x;
		mult *= 0.5;
		uv *= 2.0;
	}
	return h * TERRAIN_HEIGHT + TERRAIN_LIFT;	
}

float lookup_Normal( in vec3 pos )
{
	vec2 uv = pos.xz;
	uv *= TERRAIN_SAMPLE_SCALE;
	float h = 0.0;
	float mult = 1.0;
	for (int i = 0; i < 9; i ++)
	{
		h += mult * texture(t_PerlinNoise, uv + 0.5).x;
		mult *= 0.5;
		uv *= 2.0;
	}
	return h * TERRAIN_HEIGHT + TERRAIN_LIFT; 
}

vec3 terrain_calcNormal( vec3 pos, float t )
{
	vec3 eps = vec3(uEPS_intersect, 0.0, 0.0);
	
	return vec3( lookup_Normal(pos-eps.xyy) - lookup_Normal(pos+eps.xyy),
		     eps.x * 2.0,
		     lookup_Normal(pos-eps.yyx) - lookup_Normal(pos+eps.yyx) ) ;
}

bool isLightSourceVisible( vec3 pos, vec3 n, vec3 dirToLight)
{
	pos += n;
	float h = 1.0;
	float t = 0.0;
	float terrainHeight = TERRAIN_HEIGHT * 2.0 + TERRAIN_LIFT;

	for(int i = 0; i < 300; i++)
	{
		h = pos.y - lookup_Heightmap(pos);
		if ( pos.y > terrainHeight || h < 0.1) break;
		pos += dirToLight * h;
	}
	return h > 0.1;
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
float SceneIntersectTerrain( int checkWater )
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
	vec3 normal;
	float d = 0.0;
	float dp = 0.0;
	float t = INFINITY;
	vec3 hitPos;
	float terrainHeight;
	float waterWaveHeight;
	// Terrain
	vec3 pos = rayOrigin;
	vec3 dir = rayDirection;
	float h = 0.0;
	
	for (int i = 0; i < 300; i++)
	{
		h = pos.y - lookup_Heightmap(pos);
		if (d > TERRAIN_DETAIL_FAR || h < 1.0) break;
		d += h * 0.45;
		pos += dir * h * 0.45; 
	}
	hitPos = pos;
	if (h >= 1.0) d = INFINITY;

	if (d > TERRAIN_DETAIL_FAR)
	{
		dp = PlaneIntersect( vec4(0, 1, 0, uWaterLevel), rayOrigin, rayDirection );
		if (dp < d)
		{
			hitPos = rayOrigin + rayDirection * dp;
			terrainHeight = lookup_Heightmap(hitPos);
			d = DisplacementBoxIntersect( vec3(-INFINITY, -INFINITY, -INFINITY), vec3(INFINITY, terrainHeight, INFINITY), rayOrigin, rayDirection);
		}
		
	}
	if (d < t)
	{
		t = d;
		hitNormal = terrain_calcNormal(hitPos, t);
		hitEmission = vec3(0);
		hitColor = vec3(0);
		hitType = TERRAIN;
	}
	
	if (checkWater == FALSE)
		return t;
	
	pos = rayOrigin; // reset pos
	dir = rayDirection; // reset dir
	h = 0.0; // reset h
	d = 0.0; // reset d
	for(int i = 0; i < 50; i++)
	{
		h = abs(pos.y - getOceanWaterHeight(pos));
		if (d > WATER_DETAIL_FAR || abs(h) < 1.0) break;
		d += h;
		pos += dir * h; 
	}
	
	hitPos = pos;
	if (h >= 1.0) d = INFINITY;
	
	if (d > WATER_DETAIL_FAR)
	{
		dp = PlaneIntersect( vec4(0, 1, 0, uWaterLevel), rayOrigin, rayDirection );
		if ( dp < d )
		{
			hitPos = rayOrigin + rayDirection * dp;
			waterWaveHeight = getOceanWaterHeight_Detail(hitPos);
			d = DisplacementBoxIntersect( vec3(-INFINITY, -INFINITY, -INFINITY), vec3(INFINITY, waterWaveHeight, INFINITY), rayOrigin, rayDirection);
		}	
	}
	
	if (d < t) 
	{
		float eps = 1.0;
		t = d;
		float dx = getOceanWaterHeight_Detail(hitPos - vec3(eps,0,0)) - getOceanWaterHeight_Detail(hitPos + vec3(eps,0,0));
		float dy = eps * 2.0; // (the water wave height is a function of x and z, not dependent on y)
		float dz = getOceanWaterHeight_Detail(hitPos - vec3(0,0,eps)) - getOceanWaterHeight_Detail(hitPos + vec3(0,0,eps));
		
		hitNormal = vec3(dx,dy,dz);
		hitEmission = vec3(0);
		hitColor = vec3(0.6, 1.0, 1.0);
		hitType = REFR;
	}
		
	return t;
}


//-----------------------------------------------------------------------
vec3 CalculateRadianceTerrain()
//-----------------------------------------------------------------------
{

	vec3 initialSkyColor = Get_Sky_Color(rayDirection);
	
	vec3 skyRayOrigin = rayOrigin * vec3(0.02);
	vec3 skyRayDirection = normalize(vec3(rayDirection.x, abs(rayDirection.y), rayDirection.z));
	float dc = SphereIntersect( 20000.0, vec3(skyRayOrigin.x, -19900.0, skyRayOrigin.z) + vec3(rng() * 2.0), skyRayOrigin, skyRayDirection );
	vec3 skyPos = skyRayOrigin + skyRayDirection * dc;
	vec4 cld = render_clouds(skyPos, skyRayDirection);
	
	
	vec3 accumCol = vec3(0);
        vec3 mask = vec3(1);
	vec3 reflectionMask = vec3(1);
	vec3 reflectionRayOrigin = vec3(0);
	vec3 reflectionRayDirection = vec3(0);
	vec3 n, nl, x;
	vec3 cameraRayOrigin = rayOrigin;
	vec3 firstX = cameraRayOrigin;
	vec3 tdir;
	
	float nc, nt, ratioIoR, Re, Tr;
	//float P, RP, TP;
	float t = INFINITY;
	float thickness = 0.01;

	int previousIntersecType = -100;

	int bounceIsSpecular = TRUE;
	int checkWater = TRUE;
	int skyHit = FALSE;
	int sampleLight = FALSE;
	int willNeedReflectionRay = FALSE;
	
	//test
	vec3 randVec = vec3(rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0);
	float weight;
	float epsIntersect = 0.01;
	int diffuseCount = 0;
	hitType = -100;
	
    for (int bounces = 0; bounces < 5; bounces++)
	{
		previousIntersecType = hitType;

		float t0 = SceneIntersectViewer();
		float t1 = SceneIntersectTerrain(checkWater);
		// t = min(t0, t1);

		// if (t0!=INFINITY){
		if (t0 < t1){
			// t = SceneIntersectTerrain(checkWater);
			t = SceneIntersectViewer();
		} else {
			t = t1;
			// vec3 objectNormal = vec3(0);
			// vec3 objectColor = vec3(0);
			// float objectID = -INFINITY;
			// float pixelSharpness = 0.0;
			// CalculateRadianceViewer(objectNormal, objectColor, objectID, pixelSharpness);
			// continue;
		}
		checkWater = FALSE;

		if (t == INFINITY)
		{
			// ray hits sky first
			// if (bounces == 0)
			// {
			// 	pixelSharpness = 1.01;

			// 	accumCol += Get_HDR_Color(rayDirection);
			// 	break;
			// }


			if (bounces == 0) // ray hits sky first	
			{
				skyHit = TRUE;
				accumCol += initialSkyColor;
				break; // exit early	
			}

			if (bounceIsSpecular == TRUE)
			{
				accumCol += mask * Get_Sky_Color(rayDirection);
			}
			
			// if ray bounced off of diffuse material and hits sky
			if (previousIntersecType == DIFF)
			{
				if (sampleLight == TRUE)
					accumCol += mask * uSunColor * uSunLightIntensity * 0.5;
				else
	// vec3 initialSkyColor = Get_Sky_Color(rayDirection); Get_HDR_Color(rayDirection)
					accumCol += mask * Get_Sky_Color(rayDirection) * uSkyLightIntensity * 0.5;
			}
	
			// if ray bounced off of glass and hits sky
			if (previousIntersecType == REFR)
			{
				if (diffuseCount == 0) // camera looking through glass, hitting the sky
				{
					// pixelSharpness = 1.01;
					mask *= Get_Sky_Color(rayDirection);
				}	
				else if (sampleLight == TRUE) // sun rays going through glass, hitting another surface
					mask *= uSunColor * uSunLightIntensity;
				else  // sky rays going through glass, hitting another surface
					mask *= Get_Sky_Color(rayDirection) * uSkyLightIntensity;

				if (bounceIsSpecular == TRUE) // prevents sun 'fireflies' on diffuse surfaces
					accumCol += mask;
			}

			if (willNeedReflectionRay == TRUE)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				
				diffuseCount = 0;
				
				continue;
			}
			// reached the sky light, so we can exit
			break;
		} // end if (t == INFINITY)
		
		
		// useful data 
		n = normalize(hitNormal);
        nl = dot(n, rayDirection) < 0.0 ? n : -n;
		x = rayOrigin + rayDirection * t;
		
		if (bounces == 0) 
			firstX = x;

			// objectNormal = nl;
			// objectColor = hitColor;
			// objectID = hitObjectID;
		
		// ray hits terrain
		if (hitType == TERRAIN)
		{
			previousIntersecType = TERRAIN;

			float rockNoise = texture(t_PerlinNoise, (0.0003 * x.xz)).x;
			vec3 rockColor0 = max(vec3(0.01), vec3(0.04, 0.01, 0.01) * rockNoise);
			vec3 rockColor1 = max(vec3(0.01), vec3(0.08, 0.07, 0.07) * rockNoise);
			vec3 snowColor = vec3(0.7);
			vec3 up = vec3(0, 1, 0);
			vec3 randomSkyVec = randomCosWeightedDirectionInHemisphere(mix(n, up, 0.9));
			vec3 skyColor = Get_Sky_Color(randomSkyVec);
			if (dot(randomSkyVec, uSunDirection) > 0.98) skyColor *= 0.01;
			vec3 sunColor = clamp( Get_Sky_Color(randomDirectionInSpecularLobe(uSunDirection, 0.1)), 0.0, 4.0 );
			float terrainLayer = clamp( (x.y + (rockNoise * 500.0) * n.y) / (TERRAIN_HEIGHT * 1.5 + TERRAIN_LIFT), 0.0, 1.0 );
			
			if (terrainLayer > 0.8 && terrainLayer > 1.0 - n.y)
				hitColor = snowColor;	
			else
				hitColor = mix(rockColor0, rockColor1, clamp(n.y, 0.0, 1.0) );
				
			mask = hitColor * skyColor; // ambient color from sky light

			bounceIsSpecular = FALSE;

			vec3 shadowRayDirection = randomDirectionInSpecularLobe(uSunDirection, 0.1);						
			if (bounces < 2 && x.y > uWaterLevel && dot(n, shadowRayDirection) > 0.1 && isLightSourceVisible(x, n, shadowRayDirection) ) // in direct sunlight
			{
				mask = hitColor * mix(skyColor, sunColor, clamp(dot(n,shadowRayDirection),0.0,1.0));	
			}

			accumCol += mask;

			if (willNeedReflectionRay == TRUE)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				continue;
			}

			break;
		}
		
		if (hitType == DIFF) // Ideal DIFFUSE reflection
		{
			// previousIntersecType = DIFF;

			// if (diffuseCount == 0)
			// 	objectColor = hitColor;

			diffuseCount++;
			// float ct = float(int(hitObjectID)%10)/10.0;
			// mask *= hitColor*vec3(0.9,1.0-ct,0.1*ct);
			// mask *= hitColor * vec3(hitUV.u,hitUV.y,1.0);
			// mask *= hitColor-0.1+0.1*hitNormal;
			if (firstX.y<420.){
				mask *= hitColor*0.02;
			}else if (firstX.y<800.){
				mask *= hitColor*vec3(0.8,0.7,0.6);
			}
			else {
				mask *= hitColor*vec3(0.8,0.4,0.4);
			}
			bounceIsSpecular = FALSE;

			if (diffuseCount == 1 && rand() < 0.5)
			{
				mask *= 2.0;
				// this branch gathers color bleeding / caustics from other surfaces hit in the future
				// choose random Diffuse sample vector
				rayDirection = randomCosWeightedDirectionInHemisphere(nl);
				rayOrigin = x + nl * epsIntersect;
				continue;
			}
			
			// this branch acts like a traditional shadowRay, checking for direct light from the Sun..
			// if it has a clear path and hits the Sun on the next bounce, sunlight is gathered, otherwise returns black (shadow)
			rayDirection = normalize(uSunDirection + (randVec * 0.01));
			rayOrigin = x + nl * epsIntersect;

			weight = max(0.0, dot(rayDirection, nl));
			mask *= diffuseCount == 1 ? 2.0 : 1.0;
			mask *= weight;
			
			sampleLight = TRUE;
			continue;
			
		} // end if (hitType == DIFF)

		if (hitType == SPEC)  // Ideal SPECULAR reflection
		{
			mask *= hitColor;

			rayDirection = reflect(rayDirection, nl);
			rayOrigin = x + rayDirection * epsIntersect;

			//bounceIsSpecular = TRUE; // turn on mirror caustics
			continue;
		}

		if (hitType == REFR)  // Ideal dielectric REFRACTION
		{
			previousIntersecType = REFR;

			nc = 1.0; // IOR of air
			nt = 1.33; // IOR of water
			Re = calcFresnelReflectance(rayDirection, n, nc, nt, ratioIoR);
			Tr = 1.0 - Re;

			if (bounces == 0)
			{
				reflectionMask = mask * Re;
				reflectionRayDirection = reflect(rayDirection, nl); // reflect ray from surface
				reflectionRayOrigin = x + nl * uEPS_intersect;
				willNeedReflectionRay = TRUE;
			}

			if (Re == 1.0)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				continue;
			}
			
			// transmit ray through surface
			mask *= Tr;
			mask *= hitColor;
			
			tdir = refract(rayDirection, nl, ratioIoR);
			rayDirection = tdir;
			rayOrigin = x - nl * uEPS_intersect;
			
			continue;
			
		} // end if (hitType == REFR)
		
	} // end for (int bounces = 0; bounces < 3; bounces++)
	
	
	// atmospheric haze effect (aerial perspective)
	float hitDistance;
	
	if ( skyHit == TRUE ) // sky and clouds
	{
		vec3 cloudColor = cld.rgb / (cld.a + 0.00001);
		vec3 sunColor = clamp( Get_Sky_Color(randomDirectionInSpecularLobe(uSunDirection, 0.1)), 0.0, 5.0 );
		
		cloudColor *= sunColor;
		cloudColor = mix(initialSkyColor, cloudColor, clamp(cld.a, 0.0, 1.0));
		
		hitDistance = distance(skyRayOrigin, skyPos);
		accumCol = mask * mix( accumCol, cloudColor, clamp( exp2( -hitDistance * 0.004 ), 0.0, 1.0 ) );
	}	
	else // terrain and other objects
	{
		hitDistance = distance(cameraRayOrigin, firstX);
		accumCol = mix( initialSkyColor, accumCol, clamp( exp2( -log(hitDistance * 0.00006) ), 0.0, 1.0 ) );
		// underwater fog effect
		hitDistance = distance(cameraRayOrigin, firstX);
		hitDistance *= uCameraUnderWater;
		accumCol = mix( vec3(0.0,0.05,0.05), accumCol, clamp( exp2( -hitDistance * 0.001 ), 0.0, 1.0 ) );
	}
	
	
	return max(vec3(0), accumCol); // prevents black spot artifacts appearing in the water 
	      
}

//#include <pathtracing_main>

// tentFilter from Peter Shirley's 'Realistic Ray Tracing (2nd Edition)' book, pg. 60
float tentFilter(float x) // input: x: a random float(0.0 to 1.0), output: a filtered float (-1.0 to +1.0)
{
	return (x < 0.5) ? sqrt(2.0 * x) - 1.0 : 1.0 - sqrt(2.0 - (2.0 * x));
}

void main( void )
{
	vec3 camRight   = vec3( uCameraMatrix[0][0],  uCameraMatrix[0][1],  uCameraMatrix[0][2]);
	vec3 camUp      = vec3( uCameraMatrix[1][0],  uCameraMatrix[1][1],  uCameraMatrix[1][2]);
	vec3 camForward = vec3(-uCameraMatrix[2][0], -uCameraMatrix[2][1], -uCameraMatrix[2][2]);
	// the following is not needed - three.js has a built-in uniform named cameraPosition
	//vec3 camPos   = vec3( uCameraMatrix[3][0],  uCameraMatrix[3][1],  uCameraMatrix[3][2]);

	// calculate unique seed for rng() function
	seed = uvec2(uFrameCounter, uFrameCounter + 1.0) * uvec2(gl_FragCoord);
	// initialize rand() variables
	counter = -1.0; // will get incremented by 1 on each call to rand()
	channel = 0; // the final selected color channel to use for rand() calc (range: 0 to 3, corresponds to R,G,B, or A)
	randNumber = 0.0; // the final randomly-generated number (range: 0.0 to 1.0)
	randVec4 = vec4(0); // samples and holds the RGBA blueNoise texture value for this pixel
	randVec4 = texelFetch(tBlueNoiseTexture, ivec2(mod(floor(gl_FragCoord.xy) + floor(uRandomVec2 * 256.0), 256.0)), 0);
	
	// rand() produces higher FPS and almost immediate convergence, but may have very slight jagged diagonal edges on higher frequency color patterns, i.e. checkerboards.
	// rng() has a little less FPS on mobile, and a little more noisy initially, but eventually converges on perfect anti-aliased edges - use this if 'beauty-render' is desired.
	vec2 pixelOffset;
	if (uFrameCounter < 150.0) 
		pixelOffset = vec2( tentFilter(rand()), tentFilter(rand()) );
	else 
		pixelOffset = vec2( tentFilter(rng()), tentFilter(rng()) );
	
	// we must map pixelPos into the range -1.0 to +1.0: (-1.0,-1.0) is bottom-left screen corner, (1.0,1.0) is top-right
	vec2 pixelPos = ((gl_FragCoord.xy + vec2(0.5) + pixelOffset) / uResolution) * 2.0 - 1.0;

	vec3 rayDir = uUseOrthographicCamera ? camForward :
		      normalize( pixelPos.x * camRight * uULen + pixelPos.y * camUp * uVLen + camForward ); 
					       
	// depth of field
	vec3 focalPoint = uFocusDistance * rayDir;
	float randomAngle = rng() * TWO_PI; // pick random point on aperture
	float randomRadius = rng() * uApertureSize;
	vec3  randomAperturePos = ( cos(randomAngle) * camRight + sin(randomAngle) * camUp ) * sqrt(randomRadius);
	// point on aperture to focal point
	vec3 finalRayDir = normalize(focalPoint - randomAperturePos);

	rayOrigin = cameraPosition + randomAperturePos;
	rayOrigin += !uUseOrthographicCamera ? vec3(0) : 
		     (camRight * pixelPos.x * uULen * 100.0) + (camUp * pixelPos.y * uVLen * 100.0);
					     
	rayDirection = finalRayDir;
	

	SetupScene();

	// Edge Detection - don't want to blur edges where either surface normals change abruptly (i.e. room wall corners), objects overlap each other (i.e. edge of a foreground sphere in front of another sphere right behind it),
	// or an abrupt color variation on the same smooth surface, even if it has similar surface normals (i.e. checkerboard pattern). Want to keep all of these cases as sharp as possible - no blur filter will be applied.
	vec3 objectNormal = vec3(0);
	vec3 objectColor = vec3(0);
	float objectID = -INFINITY;
	float pixelSharpness = 0.0;

	// perform path tracing and get resulting pixel color
	//  vec4 currentPixel = vec4( vec3(CalculateRadianceViewer(objectNormal, objectColor, objectID, pixelSharpness)), 0.0 );
	vec4 currentPixel = vec4( vec3(CalculateRadianceTerrain()), 0.0 );

	// if difference between normals of neighboring pixels is less than the first edge0 threshold, the white edge line effect is considered off (0.0)
	float edge0 = 0.2; // edge0 is the minimum difference required between normals of neighboring pixels to start becoming a white edge line
	// any difference between normals of neighboring pixels that is between edge0 and edge1 smoothly ramps up the white edge line brightness (smoothstep 0.0-1.0)
	float edge1 = 0.6; // once the difference between normals of neighboring pixels is >= this edge1 threshold, the white edge line is considered fully bright (1.0)
	float difference_Nx = fwidth(objectNormal.x);
	float difference_Ny = fwidth(objectNormal.y);
	float difference_Nz = fwidth(objectNormal.z);

	float normalDifference = smoothstep(edge0, edge1, difference_Nx) + smoothstep(edge0, edge1, difference_Ny) + smoothstep(edge0, edge1, difference_Nz);
	float objectDifference = min(fwidth(objectID), 1.0);
	float colorDifference = (fwidth(objectColor.r) + fwidth(objectColor.g) + fwidth(objectColor.b)) > 0.0 ? 1.0 : 0.0;

	vec4 previousPixel = texelFetch(tPreviousTexture, ivec2(gl_FragCoord.xy), 0);

	if ( uSceneIsDynamic ) // static
	{
		if (uFrameCounter == 1.0) // camera just moved after being still
		{
			previousPixel.rgb *= (1.0 / uPreviousSampleCount) * 0.5; // essentially previousPixel *= 0.5, like below
			previousPixel.a = 0.0;
			currentPixel.rgb *= 0.5;
		}
		else if (uCameraIsMoving) // camera is currently moving
		{
			previousPixel.rgb *= 0.5; // motion-blur trail amount (old image)
			previousPixel.a = 0.0;
			currentPixel.rgb *= 0.5; // brightness of new image (noisy)
		} 
	} else // dynamic
	{
		if (uCameraIsMoving) 
		{
			previousPixel.rgb *= (1.0 / uPreviousSampleCount) * 0.8; // essentially previousPixel *= 0.5, like below
			previousPixel.a = 0.0;
			currentPixel.rgb *= 0.2;
		}
		else 
		{
			previousPixel.rgb *= 0.9; // motion-blur trail amount (old image)
			previousPixel.a = 0.0;
			currentPixel.rgb *= 0.1; // brightness of new image (noisy)
		}
	}

	// if current raytraced pixel didn't return any color value, just use the previous frame's pixel color
	if (currentPixel.rgb == vec3(0.0))
	{
		currentPixel.rgb = previousPixel.rgb;
		previousPixel.rgb *= 0.5;
		currentPixel.rgb *= 0.5;
	}


	if (colorDifference >= 1.0 || normalDifference >= 1.0 || objectDifference >= 1.0)
		pixelSharpness = 1.01;


	currentPixel.a = 1.01;//pixelSharpness;

	// Eventually, all edge-containing pixels' .a (alpha channel) values will converge to 1.01, which keeps them from getting blurred by the box-blur filter, thus retaining sharpness.
	if (previousPixel.a == 1.01)
		currentPixel.a = 1.01;

	pc_fragColor = vec4(previousPixel.rgb + currentPixel.rgb, currentPixel.a);
}
/*

void main( void )
{

	vec3 camRight   = vec3( uCameraMatrix[0][0],  uCameraMatrix[0][1],  uCameraMatrix[0][2]);
	vec3 camUp      = vec3( uCameraMatrix[1][0],  uCameraMatrix[1][1],  uCameraMatrix[1][2]);
	vec3 camForward = vec3(-uCameraMatrix[2][0], -uCameraMatrix[2][1], -uCameraMatrix[2][2]);
	
	// calculate unique seed for rng() function
	seed = uvec2(uFrameCounter, uFrameCounter + 1.0) * uvec2(gl_FragCoord);

	// initialize rand() variables
	counter = -1.0; // will get incremented by 1 on each call to rand()
	channel = 0; // the final selected color channel to use for rand() calc (range: 0 to 3, corresponds to R,G,B, or A)
	randNumber = 0.0; // the final randomly-generated number (range: 0.0 to 1.0)
	randVec4 = vec4(0); // samples and holds the RGBA blueNoise texture value for this pixel
	randVec4 = texelFetch(tBlueNoiseTexture, ivec2(mod(gl_FragCoord.xy + floor(uRandomVec2 * 256.0), 256.0)), 0);
	
	vec2 pixelOffset = vec2( tentFilter(rand()), tentFilter(rand()) ) * 0.5;
	// we must map pixelPos into the range -1.0 to +1.0
	vec2 pixelPos = ((gl_FragCoord.xy + vec2(0.5) + pixelOffset) / uResolution) * 2.0 - 1.0;

	vec3 rayDir = uUseOrthographicCamera ? camForward : 
					       normalize( pixelPos.x * camRight * uULen + pixelPos.y * camUp * uVLen + camForward );

	// depth of field
	vec3 focalPoint = uFocusDistance * rayDir;
	float randomAngle = rng() * TWO_PI; // pick random point on aperture
	float randomRadius = rng() * uApertureSize;
	vec3  randomAperturePos = ( cos(randomAngle) * camRight + sin(randomAngle) * camUp ) * sqrt(randomRadius);
	// point on aperture to focal point
	vec3 finalRayDir = normalize(focalPoint - randomAperturePos);
    
	rayOrigin = uUseOrthographicCamera ? cameraPosition + (camRight * pixelPos.x * uULen * 100.0) + (camUp * pixelPos.y * uVLen * 100.0) + randomAperturePos :
					     cameraPosition + randomAperturePos;
	rayDirection = finalRayDir;


	SetupScene(); 

	// perform path tracing and get resulting pixel color
	vec3 pixelColor = CalculateRadianceTerrain();
	
	vec3 previousColor = texelFetch(tPreviousTexture, ivec2(gl_FragCoord.xy), 0).rgb;
	
	if ( uCameraIsMoving )
	{
		previousColor *= 0.8; // motion-blur trail amount (old image)
		pixelColor *= 0.2; // brightness of new image (noisy)
	}
	else
	{
		previousColor *= 0.9; // motion-blur trail amount (old image)
		pixelColor *= 0.1; // brightness of new image (noisy)
	}

	// if current raytraced pixel didn't return any color value, just use the previous frame's pixel color
	if (pixelColor == vec3(0.0))
	{
		pixelColor = previousColor;
		previousColor *= 0.5;
		pixelColor *= 0.5;
	}
	
	
	pc_fragColor = vec4( pixelColor + previousColor, 1.01 );	
}
*/
