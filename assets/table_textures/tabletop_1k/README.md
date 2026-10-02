# Curated 1K tabletop materials

This directory contains 33 original-resolution 1K RGB maps from Poly Haven.
Every image asset is CC0, was selected from the repository's prepared
1000-entry source manifest, and was manually screened as a plausible tileable
tabletop finish. The preparation step independently verifies the official HTTPS
source URL, MD5, byte count, JPEG decode, RGB mode, and 1024 x 1024 pixel size.

The bank intentionally excludes Poly Haven model diffuse maps, outdoor
terrain, walls, roofs, props, food, and furniture photographs. The renderer
uses it only for the manipulation table; the broader 256 px bank remains the
source for room floors and walls.

The bank ships prepared: `manifest.json` records each image's source URL, MD5, SHA-256 and
byte count, and the renderer re-validates every entry against the shipped files at startup.

`manifest.json` records only the fields needed for rendering and provenance:
the Poly Haven page, direct source URL, source MD5, repository SHA-256, byte
count, decoded image signature, MuJoCo material parameters, and image license. It
does not redistribute Poly Haven tags, categories, author records, or other API
metadata that are outside the image asset's CC0 grant. Renderer startup
revalidates the local file's SHA-256, byte count, decoded signature, source URL,
and license. MuJoCo's native renderer consumes only the RGB map and legacy
reflectance / shininess / specular scalars; it does not consume PBR normal,
roughness, metallic, or AO maps.
